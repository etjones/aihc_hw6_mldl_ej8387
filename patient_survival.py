#! /usr/bin/env python3
"""
Patient survival time prediction based on MIMIC-III admissions data.
"""

import sqlite_utils
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import sys
import shap  # Add SHAP library
import os
import pickle  # For saving/loading SHAP values

# Constants
DB_PATH = "../mimic3.db"  # Adjust path as necessary
TABLE_NAME = "admissions"
TARGET_COLUMN = "SURVIVAL_DAYS"


def load_data(db_path: str) -> tuple[pd.DataFrame, dict]:
    """Loads data from the MIMIC-III database via SQLite, performs joins,
    feature engineering, and returns the final DataFrame and ICD-9 code mapping."""
    print(f"Loading data from {db_path}...")
    db = sqlite_utils.Database(db_path)

    # Load tables using the helper function
    data = {}
    data["admissions"] = _load_and_prepare_table(
        db, "admissions", date_cols=["ADMITTIME", "DISCHTIME", "DEATHTIME"]
    )
    data["patients"] = _load_and_prepare_table(db, "patients", date_cols=["DOB"])
    data["diagnoses_icd"] = _load_and_prepare_table(
        db, "diagnoses_icd", string_cols=["ICD9_CODE"]
    )

    data["d_icd_diagnoses"] = _load_and_prepare_table(
        db, "d_icd_diagnoses", string_cols=["ICD9_CODE"]
    )

    # Check if essential tables are loaded
    if data["admissions"].empty or data["patients"].empty:
        print(
            "Error: Essential tables (admissions, patients) could not be loaded. Exiting."
        )
        sys.exit(1)

    # Create ICD-9 mapping; {icd_code: short_diagnosis_title}
    icd9_map = {}
    if not data["d_icd_diagnoses"].empty:
        icd9_map = pd.Series(
            data["d_icd_diagnoses"]["SHORT_TITLE"].values,
            index=data["d_icd_diagnoses"]["ICD9_CODE"],
        ).to_dict()
    else:
        print(
            "Warning: d_icd_diagnoses table is empty. ICD-9 descriptions will be unavailable."
        )

    # --- Joins and Feature Engineering --- #

    # 1. Merge Admissions and Patients (Core join)
    df = pd.merge(
        data["admissions"],
        data["patients"][["SUBJECT_ID", "GENDER", "DOB"]],
        on="SUBJECT_ID",
        how="left",
    )

    # Call helper functions for subsequent steps
    df = _calculate_age(df)
    df = _get_primary_diagnosis(df, data["diagnoses_icd"])
    df = _calculate_survival_days(df)

    print(f"Data loading and feature engineering complete. Final shape: {df.shape}")
    # print("Final DataFrame columns:", df.columns.tolist())
    # print(df.head())
    # print(df.info())
    # print(df.describe(include='all'))

    return df, icd9_map


def _load_and_prepare_table(
    db: sqlite_utils.Database,
    table_name: str,
    date_cols: list[str] | None = None,
    string_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Loads a table, converts specified columns, and returns a DataFrame."""
    try:
        table = db[table_name]
        # Convert table rows to DataFrame
        df = pd.DataFrame(table.rows)

        if df.empty:
            print(f"Warning: Table '{table_name}' is empty.")
            return df

        # Convert date columns
        if date_cols:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert string columns
        if string_cols:
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

    except Exception as e:
        print(f"Error loading or preparing table {table_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    return df


def _calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates age at admission robustly, handling implausible values and overflows."""
    df["AGE"] = np.nan  # Initialize AGE column

    valid_dates_mask = (
        df["ADMITTIME"].notna()
        & df["DOB"].notna()
        & (df["DOB"].dt.year >= 1900)  # Exclude likely placeholder DOBs
    )

    if valid_dates_mask.any():
        # Select the relevant rows
        admittime = df.loc[valid_dates_mask, "ADMITTIME"]
        dob = df.loc[valid_dates_mask, "DOB"]

        # Calculate difference in years
        years_diff = admittime.dt.year - dob.dt.year

        # Adjust age if birthday hasn't occurred yet in the admission year
        # Correction = 1 if admit_day_of_year < dob_day_of_year, else 0
        correction = (admittime.dt.dayofyear < dob.dt.dayofyear).astype(int)

        # Calculate final age in years
        calculated_age = years_diff - correction

        # Apply the calculated age to the DataFrame subset
        df.loc[valid_dates_mask, "AGE"] = calculated_age

        # Cap age at 90 (MIMIC obfuscates ages > 89)
        # Apply clipping to the entire column after calculations
        df["AGE"] = df["AGE"].clip(upper=90)

    print(f"Calculated AGE. Missing values: {df['AGE'].isna().sum()}. Capped at 90.")
    return df


def _get_primary_diagnosis(
    df: pd.DataFrame, diagnoses_icd: pd.DataFrame
) -> pd.DataFrame:
    """Gets the primary diagnosis code (sequence 1)."""
    if not diagnoses_icd.empty:
        primary_diag = diagnoses_icd[diagnoses_icd["SEQ_NUM"] == 1][
            ["HADM_ID", "ICD9_CODE"]
        ].rename(columns={"ICD9_CODE": "PRIMARY_DIAGNOSIS_CODE"})

        df = pd.merge(df, primary_diag, on="HADM_ID", how="left")
        print(
            f"Merged PRIMARY_DIAGNOSIS_CODE. Missing values: {df['PRIMARY_DIAGNOSIS_CODE'].isna().sum()}"
        )
    else:
        df["PRIMARY_DIAGNOSIS_CODE"] = pd.NA
        print("Skipped merging PRIMARY_DIAGNOSIS_CODE due to empty input table.")
    return df


def _calculate_survival_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the target variable SURVIVAL_DAYS."""
    # Calculate based on DEATHTIME first
    died_mask = df["DEATHTIME"].notna()
    df.loc[died_mask, "SURVIVAL_DAYS"] = (
        df.loc[died_mask, "DEATHTIME"] - df.loc[died_mask, "ADMITTIME"]
    ).dt.days

    # For those who didn't die in hospital, use DISCHTIME
    survived_mask = df["DEATHTIME"].isna()
    df.loc[survived_mask, "SURVIVAL_DAYS"] = (
        df.loc[survived_mask, "DISCHTIME"] - df.loc[survived_mask, "ADMITTIME"]
    ).dt.days

    # Handle cases where ADMITTIME or DISCHTIME might be NaT
    df["SURVIVAL_DAYS"] = df["SURVIVAL_DAYS"].fillna(
        0
    )  # Or another imputation strategy
    # Ensure non-negative survival days
    df["SURVIVAL_DAYS"] = df["SURVIVAL_DAYS"].apply(lambda x: max(x, 0))
    print(
        f"Calculated SURVIVAL_DAYS. Missing values: {df['SURVIVAL_DAYS'].isna().sum()}"
    )
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Preprocesses the data (already joined), calculates target, and sets up transformers."""
    print("Preprocessing data...")

    # Convert date columns to datetime objects (if not already done, safe to repeat)
    # Note: DOB is not needed here anymore as AGE is calculated
    date_columns = ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"]
    for col in date_columns:
        if col in df.columns:  # Check if column exists
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Calculate target variable: Survival time in days
    # Assign NaN if DEATHTIME is NaT (Not a Time)
    if "DEATHTIME" in df.columns and "ADMITTIME" in df.columns:
        df[TARGET_COLUMN] = (df["DEATHTIME"] - df["ADMITTIME"]).dt.total_seconds() / (
            24 * 60 * 60
        )  # Convert to days
    else:
        # Handle case where necessary columns for target are missing
        raise ValueError("Missing DEATHTIME or ADMITTIME, cannot calculate target.")

    # --- Feature Engineering / Selection (within preprocess) ---
    # REMOVED: LOS calculation - deemed potentially leaky/too correlated with target
    # if 'DISCHTIME' in df.columns and 'ADMITTIME' in df.columns:
    #     df["LOS"] = (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / (
    #         24 * 60 * 60
    #     )
    # else:
    #     df["LOS"] = np.nan

    # Drop original date columns and identifiers not used as direct features
    # Also drop DEATHTIME as it's directly used to calculate the target
    # Drop DISCHTIME as LOS captures similar info related to ADMITTIME
    # Drop HOSPITAL_EXPIRE_FLAG and DISCHARGE_LOCATION as they leak outcome information
    columns_to_drop = [
        "ROW_ID",
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "DEATHTIME",
        "DOB",  # DOB dropped as AGE is used
        "EDREGTIME",
        "EDOUTTIME",
        "DIAGNOSIS",  # DIAGNOSIS is free text
        "HOSPITAL_EXPIRE_FLAG",
        "DISCHARGE_LOCATION",  # Outcome leakage (already dropped)
    ]
    # Use errors='ignore' in case a column was already dropped or doesn't exist
    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns], errors="ignore"
    )

    # Separate features (X) and target (y)
    # Drop rows where the target is NaN (patient didn't die or DEATHTIME was NaT)
    print(f"Shape before dropping NaNs in target: {df.shape}")
    df_target_known = df.dropna(subset=[TARGET_COLUMN]).copy()
    print(f"Shape after dropping NaNs in target: {df_target_known.shape}")
    if df_target_known.empty:
        raise ValueError(
            "No data available after removing rows with missing target values."
        )

    y = df_target_known[TARGET_COLUMN]
    X = df_target_known.drop(columns=[TARGET_COLUMN])

    # Identify categorical and numerical features (re-identify after dropping columns and adding new ones)
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    # Explicitly add known categorical even if inferred as numeric sometimes
    known_categorical = [
        "PRIMARY_DIAGNOSIS_CODE",
        "FIRST_CAREUNIT",
        "GENDER",
        "ADMISSION_TYPE",
        "ADMISSION_LOCATION",
        "INSURANCE",
        "LANGUAGE",
        "RELIGION",
        "MARITAL_STATUS",
        "ETHNICITY",
    ]
    for col in known_categorical:
        if col in X.columns and col not in categorical_features:
            categorical_features.append(col)
        # Ensure these are treated as strings/objects for consistent handling
        if col in X.columns:
            X[col] = X[col].astype(str)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Explicitly ensure new numeric features are included
    known_numerical = [
        "AGE",
        "ICU_ADMISSION_FLAG",
    ]  # LOS, removed
    for col in known_numerical:
        if col in X.columns and col not in numerical_features:
            numerical_features.append(col)

    # Ensure no overlap and remove any non-feature columns that slipped through
    categorical_features = [
        col
        for col in categorical_features
        if col in X.columns
        and col
        in known_categorical + ["LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY"]
    ]  # Refine list
    numerical_features = [
        col
        for col in numerical_features
        if col in X.columns and col not in categorical_features
    ]

    print(f"Final Categorical features for preprocessing: {categorical_features}")
    print(f"Final Numerical features for preprocessing: {numerical_features}")

    # --- Preprocessing Steps ---
    transformers = []
    if numerical_features:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numerical_transformer, numerical_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),  # Ensure dense output
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any columns not explicitly handled
    )

    # Fit the preprocessor and transform the data
    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X)

    print(f"Preprocessing complete. Processed feature shape: {X_processed.shape}")
    # Return the processed DataFrame, target Series, and the *fitted* preprocessor
    return X_processed, y, preprocessor


class PatientDataset(Dataset):
    """PyTorch Dataset for patient admission data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Ensure labels are the correct shape [n_samples, 1] for regression loss (like MSELoss)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            if labels.ndim == 1
            else torch.tensor(labels, dtype=torch.float32)
        )
        if self.labels.ndim == 1:
            self.labels = self.labels.unsqueeze(1)
        elif self.labels.shape[1] != 1:
            # If labels are already 2D, ensure the second dimension is 1
            # This case might occur if labels were somehow processed into multiple columns
            # For standard regression, we usually expect a single target value per sample.
            # If this happens, it indicates an issue upstream.
            print(
                f"Warning: Labels tensor has shape {self.labels.shape}. For regression, expected shape [n_samples, 1]. Check data processing."
            )
            # Attempt to reshape or select the first column if appropriate, but this is risky.
            # self.labels = self.labels[:, 0].unsqueeze(1) # Example: select first column

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the features and label for a given index."""
        return self.features[idx], self.labels[idx]


def get_feature_names(column_transformer: ColumnTransformer) -> list[str]:
    """Gets feature names from a ColumnTransformer."""
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(pipe, "get_feature_names_out"):
            # For sklearn >= 0.24
            feature_names = pipe.get_feature_names_out(features)
            output_features.extend(feature_names)
        elif hasattr(pipe, "get_feature_names"):
            feature_names = pipe.get_feature_names()
            output_features.extend(feature_names)
        else:
            # For simple transformers or if names don't change
            output_features.extend(features)

    # Debug information
    print(f"Number of features after preprocessing: {len(feature_names)}")
    print(f"First 10 feature names: {feature_names[:10]}")

    return output_features


class SurvivalPredictor(nn.Module):
    """Simple MLP for survival time prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(
            32, 1
        )  # Predicting a single value (survival days)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> tuple[nn.Module, list[float], list[float]]:
    """Trains the PyTorch model."""
    print("Starting training...")
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_features)

            # Calculate loss
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_epoch_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                epoch_val_loss += loss.item()

        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}"
        )

    # TODO: Implement training loop (epochs, batches, backpropagation) - Done
    # TODO: Implement validation loop - Done
    # TODO: Record training and validation loss per epoch - Done

    print("Training finished.")
    return model, train_losses, val_losses


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train a model for a specified number of epochs and track losses.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimization

    Returns:
        Tuple containing:
        - Trained model
        - List of training losses per epoch
        - List of validation losses per epoch
    """
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model, epoch_train_losses, epoch_val_losses = train_model(
            model, train_loader, val_loader, epochs=1, learning_rate=learning_rate
        )
        train_loss = epoch_train_losses[0]
        val_loss = epoch_val_losses[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    return model, train_losses, val_losses


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates the model on a given data loader."""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        # TODO: Implement evaluation logic - Done basic loss calc
        pass  # Removed placeholder
    # Return average loss per batch
    return total_loss / len(loader)


def plot_training_validation_loss(
    train_losses, val_losses, output_path="assets/training_validation_loss.png"
):
    """Plot and save training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.savefig(output_path)
    print(f"Saved training/validation loss plot to {output_path}")


def _get_feature_names_from_preprocessor(preprocessor, X):
    """Get feature names from the preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer
        X: Original features DataFrame

    Returns:
        List of feature names after preprocessing
    """
    # Get feature names from the preprocessor
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # For older scikit-learn versions
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                # For sklearn >= 0.24
                feature_names.extend(transformer.get_feature_names_out())
            elif hasattr(transformer, "get_feature_names"):
                feature_names.extend(transformer.get_feature_names())
            else:
                # For simple transformers or if names don't change
                feature_names.extend(columns)

    # Debug information
    print(f"Number of features after preprocessing: {len(feature_names)}")
    print(f"First 10 feature names: {feature_names[:10]}")

    return feature_names


def _create_descriptive_feature_names(encoded_feature_names, icd9_map):
    """Create more descriptive feature names for visualization.

    Args:
        encoded_feature_names: Feature names from the preprocessor
        icd9_map: Dictionary mapping ICD9 codes to descriptions

    Returns:
        List of descriptive feature names
    """
    descriptive_names = []

    for name in encoded_feature_names:
        # Handle OneHotEncoder feature names (format: encoder__feature_value)
        if "__" in name:
            parts = name.split("__")
            if len(parts) >= 2:
                feature = parts[0].replace("onehotencoder", "").strip("_")
                value = parts[1]

                # Special handling for ICD9 codes
                if feature == "PRIMARY_DIAGNOSIS_CODE" and value in icd9_map:
                    descriptive_names.append(f"{feature}: {value} ({icd9_map[value]})")
                else:
                    descriptive_names.append(f"{feature}: {value}")
            else:
                descriptive_names.append(name)
        else:
            descriptive_names.append(name)

    # Debug information
    print(f"Number of descriptive feature names: {len(descriptive_names)}")
    print(f"First 10 descriptive feature names: {descriptive_names[:10]}")

    return descriptive_names


def _get_permutation_feature_names(preprocessor, X_val, icd9_map):
    """Extract feature names from a fitted ColumnTransformer for permutation importance.

    Handles different sklearn versions and fallback methods.
    """
    try:
        # For sklearn >= 0.24
        encoded_names = preprocessor.get_feature_names_out()

        # Create human-readable feature names
        descriptive_names = []
        for name in encoded_names:
            if name.startswith("cat__ICD9_CODE_") or name.startswith(
                "cat__PRIMARY_DIAGNOSIS_CODE_"
            ):
                # Extract the code part
                if name.startswith("cat__ICD9_CODE_"):
                    code = name.split("cat__ICD9_CODE_")[-1]
                else:
                    code = name.split("cat__PRIMARY_DIAGNOSIS_CODE_")[-1]

                description = icd9_map.get(code, f"Unknown ICD9 ({code})")
                # Limit description length for readability
                max_len = 40
                descriptive_name = f"Diag: {description[:max_len]}{'...' if len(description) > max_len else ''}"
                descriptive_names.append(descriptive_name)
            elif name.startswith("num__"):
                descriptive_names.append(name.split("num__")[-1])
            elif name.startswith("cat__"):
                # Handle other categorical features like FIRST_CAREUNIT
                descriptive_names.append(name.split("cat__")[-1])
            else:
                descriptive_names.append(name)  # Keep original name if no prefix

        return encoded_names, descriptive_names

    except AttributeError:
        # Fallback: Manual construction (less robust, assumes specific structure)
        print(
            "Warning: preprocessor does not have get_feature_names_out. Attempting manual construction."
        )
        num_features_info = preprocessor.transformers_[
            0
        ]  # ('num', StandardScaler(), [num_cols])
        cat_features_info = preprocessor.transformers_[
            1
        ]  # ('cat', OneHotEncoder(...), [cat_cols])
        num_features = num_features_info[2]
        cat_transformer = cat_features_info[1]
        cat_features = cat_features_info[2]

        # Get numeric feature names
        numeric_feature_names = list(
            X_val.select_dtypes(include=np.number).columns[num_features]
        )

        # Get categorical feature names
        if hasattr(cat_transformer, "get_feature_names_out"):
            cat_feature_names_out = cat_transformer.get_feature_names_out(cat_features)
        elif hasattr(cat_transformer, "get_feature_names"):  # Older sklearn
            cat_feature_names_out = cat_transformer.get_feature_names(cat_features)
        else:
            print("Error: Cannot extract feature names from categorical transformer.")
            return [], []

        encoded_names = list(numeric_feature_names) + list(cat_feature_names_out)

        # Create human-readable feature names (same as above)
        descriptive_names = []
        for name in encoded_names:
            if "ICD9_CODE_" in name or "PRIMARY_DIAGNOSIS_CODE_" in name:
                # Extract the code part
                if "ICD9_CODE_" in name:
                    code = name.split("ICD9_CODE_")[-1]
                else:
                    code = name.split("PRIMARY_DIAGNOSIS_CODE_")[-1]

                description = icd9_map.get(code, f"Unknown ICD9 ({code})")
                # Limit description length for readability
                max_len = 40
                descriptive_name = f"Diag: {description[:max_len]}{'...' if len(description) > max_len else ''}"
                descriptive_names.append(descriptive_name)
            else:
                descriptive_names.append(name)  # Keep original name if no prefix

        return encoded_names, descriptive_names


def calculate_feature_importance(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    X_val: pd.DataFrame,  # Original validation features before preprocessing
    y_val: pd.Series,
    icd9_map: dict,  # Mapping from ICD9 code to description
    n_top_features: int = 20,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> tuple[list[str], list[float]]:
    """Calculates and plots feature importance using permutation importance."""
    print("Calculating feature importance...")
    device = get_torch_device()

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    try:
        # Get feature names from the preprocessor - use the original method for permutation importance
        encoded_feature_names, descriptive_feature_names = (
            _get_permutation_feature_names(preprocessor, X_val, icd9_map)
        )

        # Preprocess the validation data
        X_val_processed = preprocessor.transform(X_val)
    except Exception as e:
        print(
            f"Error during preprocessing or getting feature names in importance calculation: {e}"
        )
        print("Columns available in X_val:", X_val.columns.tolist())
        print("Preprocessor details:", preprocessor)
        return [], []

    # Convert to tensor
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    # Move model to the same device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    criterion = nn.MSELoss()
    importances = []

    # Get baseline loss
    with torch.no_grad():
        y_val_processed = y_val_tensor.view(-1, 1)
        baseline_preds = model(X_val_tensor)
        baseline_loss = criterion(baseline_preds, y_val_processed).item()

    # Calculate importance for each feature
    for i in range(X_val_processed.shape[1]):
        permuted_losses = []

        # Store original column for restoration
        original_column = X_val_processed[:, i].copy()

        # Repeat permutation multiple times for stability
        for _ in range(n_repeats):
            # Create permutation indices
            perm_indices = np.random.permutation(len(X_val_processed))

            # Permute one feature and keep others unchanged
            X_val_processed[:, i] = original_column[perm_indices]

            with torch.no_grad():
                permuted_loss = criterion(
                    model(
                        torch.tensor(X_val_processed, dtype=torch.float32).to(device)
                    ),
                    y_val_processed,
                ).item()
            permuted_losses.append(permuted_loss)

            # Restore original column for the next repeat or feature
            X_val_processed[:, i] = original_column

        avg_permuted_loss = np.mean(permuted_losses)
        importance = (
            avg_permuted_loss - baseline_loss
        )  # Higher value means more important
        importances.append(importance)

    # Ensure lengths match
    if len(importances) != len(descriptive_feature_names):
        print(
            f"Warning: Mismatch between importances ({len(importances)}) and "
            f"descriptive names ({len(descriptive_feature_names)}). Using encoded names."
        )
        descriptive_feature_names = encoded_feature_names  # Fallback

    # Select Top N Features based on importance scores
    importances_np = np.array(importances)
    top_n_indices_array = np.argsort(importances_np)[-n_top_features:][::-1]

    top_n_descriptive_names = [
        descriptive_feature_names[i] for i in top_n_indices_array
    ]
    top_n_importances = [importances[i] for i in top_n_indices_array]

    # Create a separate bar plot for feature importance magnitude
    plt.figure(figsize=(10, n_top_features / 2))
    # Sort features by absolute magnitude for the bar chart
    sorted_indices = np.argsort(np.abs(top_n_importances))
    sorted_features = [top_n_descriptive_names[i] for i in sorted_indices]
    sorted_importances = [top_n_importances[i] for i in sorted_indices]
    # Plot the bars
    plt.barh(range(len(sorted_features)), sorted_importances, align="center")
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel("Feature Importance (MSE increase when feature is permuted)")
    plt.title(f"Top {n_top_features} Features by Importance")
    plt.tight_layout()

    # Save the magnitude plot
    out_path_magnitude = "assets/feature_importance.png"
    plt.savefig(out_path_magnitude)
    print(f"Sorted feature importance plot saved to {out_path_magnitude}")

    # Create a DataFrame with feature impacts for easier interpretation
    # Calculate mean SHAP values for the top features
    mean_shap_values = []
    for i, idx in enumerate(top_n_indices_array):
        mean_shap = float(np.mean(top_n_importances[i]))
        mean_shap_values.append(mean_shap)

    impact_df = pd.DataFrame(
        {
            "Feature": top_n_descriptive_names,
            "Mean_Abs_SHAP": top_n_importances,
            "Mean_SHAP": mean_shap_values,
        }
    )

    # Add direction interpretation
    impact_df["Direction"] = impact_df["Mean_SHAP"].apply(
        lambda x: "Increases survival days" if x > 0 else "Decreases survival days"
    )

    print("\nFeature Impact Summary:")
    print(impact_df)

    # Save the impact summary
    impact_path = "assets/feature_impact_summary.csv"
    impact_df.to_csv(impact_path, index=False)
    print(f"Feature impact summary saved to {impact_path}")

    return top_n_descriptive_names, top_n_importances


def calculate_shap_values(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    X_val: pd.DataFrame,  # Original validation features before preprocessing
    y_val: pd.Series,
    icd9_map: dict,  # Mapping from ICD9 code to description
    n_top_features: int = 20,
    n_background_samples: int = 100,
    random_state: int | None = None,
    force_recalculate: bool = False,
) -> tuple[list[str], list[float]]:
    """Calculates and plots SHAP values to interpret feature importance with direction.

    Unlike permutation importance, SHAP values show both magnitude AND direction
    of feature impact on predictions.

    Args:
        model: Trained PyTorch model
        preprocessor: Fitted ColumnTransformer
        X_val: Original validation features (before preprocessing)
        y_val: Validation target values
        icd9_map: Dictionary mapping ICD9 codes to descriptions
        n_top_features: Number of top features to display
        n_background_samples: Number of background samples for SHAP explainer
        random_state: Random seed for reproducibility
        force_recalculate: If True, recalculate SHAP values even if cached values exist

    Returns:
        Tuple of (feature_names, shap_values) for the top features
    """
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)

    # Define checkpoint paths
    shap_checkpoint_path = "assets/shap_values_checkpoint.pkl"

    # If force_recalculate is True, delete the existing checkpoint file
    if force_recalculate and os.path.exists(shap_checkpoint_path):
        print(
            f"Force recalculate flag is set. Deleting existing checkpoint: {shap_checkpoint_path}"
        )
        os.remove(shap_checkpoint_path)

    # shap values take a long time to compute; load from disk if available
    if not force_recalculate and os.path.exists(shap_checkpoint_path):
        print("Loading SHAP values from checkpoint...")
        try:
            with open(shap_checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Recalculating SHAP values...")
            force_recalculate = True

    # If no saved checkpoint, calculate shap values and save to disk
    else:
        print("Calculating SHAP values for feature importance interpretation...")
        device = get_torch_device()

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Preprocess the validation data first
        X_val_processed = preprocessor.transform(X_val)
        print(f"X_val shape before preprocessing: {X_val.shape}")
        print(f"X_val shape after preprocessing: {X_val_processed.shape}")

        # Get feature names from the preprocessor
        encoded_feature_names = _get_feature_names_from_preprocessor(
            preprocessor, X_val
        )

        # Create descriptive names for plotting
        descriptive_feature_names = _create_descriptive_feature_names(
            encoded_feature_names, icd9_map
        )

        # Verify feature names match the processed data dimensions
        if len(encoded_feature_names) != X_val_processed.shape[1]:
            print(
                f"WARNING: Feature names count ({len(encoded_feature_names)}) doesn't match processed data shape ({X_val_processed.shape[1]})"
            )
            # Fallback to generic feature names if mismatch
            encoded_feature_names = [
                f"feature_{i}" for i in range(X_val_processed.shape[1])
            ]
            descriptive_feature_names = encoded_feature_names.copy()

        # Move model to the same device and set to evaluation mode
        model = model.to(device)
        model.eval()

        # Create a PyTorch model wrapper for SHAP
        class ModelWrapper:
            def __init__(self, model):
                self.model = model

            def __call__(self, X):
                with torch.no_grad():
                    # Convert to numpy array first if it's not already
                    if not isinstance(X, np.ndarray):
                        X = np.array(X)
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    return self.model(X_tensor).cpu().numpy()

        model_wrapper = ModelWrapper(model)

        # Sample background data for the SHAP explainer (for computational efficiency)
        if n_background_samples < X_val_processed.shape[0]:
            background_indices = np.random.choice(
                X_val_processed.shape[0], n_background_samples, replace=False
            )
            background_data = X_val_processed[background_indices]
        else:
            background_data = X_val_processed

        # Create the SHAP explainer
        # Try using KernelExplainer which works with any model
        explainer = shap.KernelExplainer(model_wrapper, background_data)

        # Calculate SHAP values for validation data
        # For efficiency, we can use a subset of validation data
        n_samples = min(
            300, X_val_processed.shape[0]
        )  # Limit to 300 samples for efficiency
        sample_indices = np.random.choice(
            X_val_processed.shape[0], n_samples, replace=False
        )
        X_val_processed_sample = X_val_processed[sample_indices]

        # Convert sample indices to integer to avoid the error
        sample_indices = sample_indices.astype(int)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_val_processed_sample)

        # If shap_values is a list (multi-output model), take the first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Debug information
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Mean absolute SHAP values shape: {mean_abs_shap.shape}")
        print(f"Top 5 mean absolute SHAP values: {mean_abs_shap[:5]}")

        # Get indices of top features by mean absolute SHAP value
        top_n_indices_array = np.argsort(mean_abs_shap)[-n_top_features:]

        # Debug information
        print(f"Top indices array: {top_n_indices_array}")

        # Convert indices to Python integers to avoid numpy array indexing issues
        top_indices = [int(i) for i in top_n_indices_array]

        # Debug information
        print(f"Top indices as integers: {top_indices}")

        # Check for unique indices
        if len(set(top_indices)) < len(top_indices):
            print("WARNING: Duplicate indices found in top_indices!")
            # Ensure unique indices
            top_indices = list(dict.fromkeys(top_indices))
            # If we don't have enough unique indices, add more
            while len(top_indices) < n_top_features:
                # Find next best index not already in top_indices
                for i in range(len(mean_abs_shap)):
                    if i not in top_indices:
                        top_indices.append(i)
                        if len(top_indices) >= n_top_features:
                            break

        # Get names and SHAP values for top features
        top_feature_names = [descriptive_feature_names[i] for i in top_indices]
        top_feature_shap = [float(mean_abs_shap[i]) for i in top_indices]

        # Save checkpoint
        checkpoint = {
            "shap_values": shap_values,
            "feature_names": encoded_feature_names,
            "descriptive_feature_names": descriptive_feature_names,
            "X_val_processed_sample": X_val_processed_sample,
            "mean_abs_shap": mean_abs_shap,
            "top_indices": top_indices,
            "top_feature_names": top_feature_names,
            "top_feature_shap": top_feature_shap,
        }

        with open(shap_checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"Saved SHAP values to checkpoint: {shap_checkpoint_path}")

    # Extract data from checkpoint
    shap_values = checkpoint["shap_values"]
    encoded_feature_names = checkpoint["feature_names"]
    descriptive_feature_names = checkpoint["descriptive_feature_names"]
    X_val_processed_sample = checkpoint["X_val_processed_sample"]
    mean_abs_shap = checkpoint["mean_abs_shap"]
    top_indices = checkpoint["top_indices"]
    top_feature_names = checkpoint["top_feature_names"]
    top_feature_shap = checkpoint["top_feature_shap"]

    print("Successfully loaded SHAP values from checkpoint.")

    # Skip to plotting with loaded values
    print("Generating plots from cached SHAP values...")

    # Debug information
    print(f"Top feature names: {top_feature_names}")
    print(f"Top feature SHAP values: {top_feature_shap}")

    # Calculate mean SHAP values for the top features
    mean_shap_values = []
    for i, idx in enumerate(top_indices):
        mean_shap = float(np.mean(shap_values[:, idx]))
        mean_shap_values.append(mean_shap)

    # Create a DataFrame with feature impacts for easier interpretation
    impact_df = pd.DataFrame(
        {
            "Feature": top_feature_names,
            "Mean_Abs_SHAP": top_feature_shap,
            "Mean_SHAP": mean_shap_values,
        }
    )

    # Add direction interpretation
    impact_df["Direction"] = impact_df["Mean_SHAP"].apply(
        lambda x: "Increases survival days" if x > 0 else "Decreases survival days"
    )

    print("\nFeature Impact Summary:")
    print(impact_df)

    # Save the impact summary
    impact_path = "assets/feature_impact_summary.csv"
    impact_df.to_csv(impact_path, index=False)
    print(f"Feature impact summary saved to {impact_path}")

    # Create a separate bar plot for feature importance magnitude
    plt.figure(figsize=(10, n_top_features / 2))
    # Sort features by absolute magnitude for the bar chart
    sorted_indices = np.argsort(np.abs(mean_shap_values))
    sorted_features = [top_feature_names[i] for i in sorted_indices]
    sorted_shap_values = [mean_shap_values[i] for i in sorted_indices]
    # Create color map based on positive/negative values
    colors = ["red" if x < 0 else "green" for x in sorted_shap_values]
    # Plot the bars
    plt.barh(
        range(len(sorted_features)),
        sorted_shap_values,
        align="center",
        color=colors,
    )
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel("Mean SHAP Value (+ increases, - decreases survival days)")
    plt.title(f"Top {n_top_features} Features by SHAP Value (Sorted by Magnitude)")
    # Add a legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Increases survival days"),
        Patch(facecolor="red", label="Decreases survival days"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()

    # Save the magnitude plot
    out_path_magnitude = "assets/shap_feature_magnitude.png"
    plt.savefig(out_path_magnitude)
    print(f"SHAP feature magnitude plot saved to {out_path_magnitude}")


def run_feature_importance_analysis(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    raw_df: pd.DataFrame,
    icd9_map: dict,
    target_column: str = "SURVIVAL_DAYS",
    force_recalculate_shap: bool = False,
):
    """Run feature importance analysis on the trained model.

    This will calculate both permutation importance and SHAP values
    for feature interpretation.

    Args:
        model: Trained PyTorch model
        preprocessor: Fitted ColumnTransformer
        raw_df: Raw dataframe with all columns
        icd9_map: Dictionary mapping ICD9 codes to descriptions
        target_column: Name of the target column
        force_recalculate_shap: If True, recalculate SHAP values even if cached values exist
    """
    # Prepare data for feature importance calculation
    X_val, y_val = _prepare_data_for_feature_importance(raw_df, target_column)

    # Calculate feature importance using permutation importance
    calculate_feature_importance(
        model=model,
        preprocessor=preprocessor,
        X_val=X_val,
        y_val=y_val,
        icd9_map=icd9_map,
    )

    # Calculate SHAP values for feature importance interpretation
    calculate_shap_values(
        model=model,
        preprocessor=preprocessor,
        X_val=X_val,
        y_val=y_val,
        icd9_map=icd9_map,
        force_recalculate=force_recalculate_shap,
    )


def _prepare_data_for_feature_importance(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for feature importance calculation.

    Args:
        df: Raw dataframe with all columns
        target_column: Name of the target column to calculate

    Returns:
        Tuple of (X_original, y_original) with features and target
    """
    # 1. Calculate TARGET_COLUMN
    if "DEATHTIME" in df.columns and "ADMITTIME" in df.columns:
        # Ensure columns are datetime
        df["DEATHTIME"] = pd.to_datetime(df["DEATHTIME"], errors="coerce")
        df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
        df[target_column] = (df["DEATHTIME"] - df["ADMITTIME"]).dt.total_seconds() / (
            24 * 60 * 60
        )
    else:
        # This should ideally not happen if raw_df was loaded correctly, but good practice
        raise ValueError(
            "Missing DEATHTIME or ADMITTIME in raw_df for feature importance setup."
        )

    # 2. Drop rows where target is NaN
    df_target_known = df.dropna(subset=[target_column]).copy()

    # 3. Separate features and target
    y_original = df_target_known[target_column]
    X_original = df_target_known.drop(columns=[target_column])

    # 4. Drop columns that are *not* features fed into the preprocessor
    #    (IDs, original dates, intermediate columns, leaky columns)
    columns_to_drop = [
        "ROW_ID",
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "DEATHTIME",
        "DOB",  # Original date/ID columns
        "EDREGTIME",
        "EDOUTTIME",
        "DIAGNOSIS",
        "HOSPITAL_EXPIRE_FLAG",
        "DISCHARGE_LOCATION",  # Leaky columns
        # "HAS_CHARTEVENTS_DATA",
    ]
    X_original = X_original.drop(
        columns=[col for col in columns_to_drop if col in X_original.columns],
        errors="ignore",
    )

    print(f"Columns for feature importance: {X_original.columns.tolist()}")
    return X_original, y_original


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Main execution function."""
    # 1. Load Data: convert tables to dataframes, and merge frames to get the
    # features we're interested in
    raw_df, icd9_map = load_data(DB_PATH)

    # 2. Preprocess Data (Calculates target, handles features, splits)
    # Use copy to avoid modifying original raw_df
    X, y, preprocessor = preprocess_data(raw_df.copy())

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to numpy arrays
    X_train_np = X_train
    X_test_np = X_test
    y_train_np = y_train.values
    y_test_np = y_test.values

    # Create PyTorch datasets
    train_dataset = PatientDataset(X_train_np, y_train_np)
    val_dataset = PatientDataset(X_test_np, y_test_np)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 4. Initialize Model
    input_dim = X_train_np.shape[1]  # Number of features after preprocessing
    model = SurvivalPredictor(input_dim=input_dim).to(get_torch_device())

    # 6. Initial Validation
    val_loss = evaluate_model(model, val_loader)
    print(f"Initial Validation Loss (MSE): {val_loss:.4f}")

    # 7. Train the model
    model, train_losses, val_losses = _train_model(
        model, train_loader, val_loader, num_epochs=10, learning_rate=0.001
    )
    print("Training finished.")
    print(f"Final Validation Loss (MSE): {val_losses[-1]:.4f}")

    # 8. Plot Training/Validation Loss
    plot_training_validation_loss(train_losses, val_losses)

    # 9. Feature Importance Analysis
    # Check if we should force recalculation of SHAP values
    force_recalculate_shap = "--force-shap" in sys.argv

    run_feature_importance_analysis(
        model=model,
        preprocessor=preprocessor,
        raw_df=raw_df,
        icd9_map=icd9_map,
        target_column=TARGET_COLUMN,
        force_recalculate_shap=force_recalculate_shap,
    )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
