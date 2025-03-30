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
    data["procedures_icd"] = _load_and_prepare_table(
        db, "procedures_icd", string_cols=["ICD9_CODE"]
    )
    data["icustays"] = _load_and_prepare_table(db, "icustays", date_cols=["INTIME"])
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
    df = _merge_diagnosis_counts(df, data["diagnoses_icd"])
    df = _merge_procedure_counts(df, data["procedures_icd"])
    # df = _merge_icu_info(df, data["icustays"])
    df = _get_primary_diagnosis(df, data["diagnoses_icd"])
    df = _calculate_survival_days(df)

    # Add HAS_CHARTEVENTS_DATA placeholder (assuming 1 if in admissions table)
    # This might need refinement based on actual CHARTEVENTS availability check if needed
    df["HAS_CHARTEVENTS_DATA"] = 1

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


def _merge_diagnosis_counts(
    df: pd.DataFrame, diagnoses_icd: pd.DataFrame
) -> pd.DataFrame:
    """Merges diagnosis counts per admission."""
    if not diagnoses_icd.empty:
        diag_counts = (
            diagnoses_icd.groupby("HADM_ID").size().reset_index(name="NUM_DIAGNOSES")
        )
        df = pd.merge(df, diag_counts, on="HADM_ID", how="left")
        df["NUM_DIAGNOSES"] = df["NUM_DIAGNOSES"].fillna(0).astype(int)
        print(
            f"Merged NUM_DIAGNOSES. Missing values: {df['NUM_DIAGNOSES'].isna().sum()}"
        )
    else:
        df["NUM_DIAGNOSES"] = 0
        print("Skipped merging NUM_DIAGNOSES due to empty input table.")
    return df


def _merge_procedure_counts(
    df: pd.DataFrame, procedures_icd: pd.DataFrame
) -> pd.DataFrame:
    """Merges procedure counts per admission."""
    if not procedures_icd.empty:
        proc_counts = (
            procedures_icd.groupby("HADM_ID").size().reset_index(name="NUM_PROCEDURES")
        )
        df = pd.merge(df, proc_counts, on="HADM_ID", how="left")
        df["NUM_PROCEDURES"] = df["NUM_PROCEDURES"].fillna(0).astype(int)
        print(
            f"Merged NUM_PROCEDURES. Missing values: {df['NUM_PROCEDURES'].isna().sum()}"
        )
    else:
        df["NUM_PROCEDURES"] = 0
        print("Skipped merging NUM_PROCEDURES due to empty input table.")
    return df


def _merge_icu_info(df: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Merges ICU admission flag and first care unit."""
    if not icustays.empty:
        # Keep only the first ICU stay per HADM_ID based on INTIME
        first_icu = icustays.loc[icustays.groupby("HADM_ID")["INTIME"].idxmin()][
            ["HADM_ID", "FIRST_CAREUNIT"]
        ]

        df = pd.merge(df, first_icu, on="HADM_ID", how="left")
        df["ICU_ADMISSION_FLAG"] = df["FIRST_CAREUNIT"].notna().astype(int)
        print(
            f"Merged ICU info. Missing FIRST_CAREUNIT: {df['FIRST_CAREUNIT'].isna().sum()}"
        )
    else:
        df["FIRST_CAREUNIT"] = pd.NA
        df["ICU_ADMISSION_FLAG"] = 0
        print("Skipped merging ICU info due to empty input table.")
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
    ]  # LOS, NUM_DIAGNOSES, NUM_PROCEDURES removed
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

    # Remove NUM_DIAGNOSES and NUM_PROCEDURES if they exist in the list
    features_to_exclude = ["NUM_DIAGNOSES", "NUM_PROCEDURES"]
    numerical_features = [f for f in numerical_features if f not in features_to_exclude]
    categorical_features = [
        f for f in categorical_features if f not in features_to_exclude
    ]
    print(
        f"Features after excluding counts: {numerical_features + categorical_features}"
    )

    # Check for empty lists
    if not numerical_features and not categorical_features:
        raise ValueError("No features identified for preprocessing.")
    if X.empty:
        raise ValueError("Feature set X is empty before creating preprocessor.")

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

    # Convert processed data back to DataFrame for easier inspection (optional)
    # Removed unused X_processed_df assignment

    print(f"Preprocessing complete. Processed feature shape: {X_processed.shape}")
    # Return the processed DataFrame, target Series, and the *fitted* preprocessor
    return X_processed, y, preprocessor


class PatientDataset(Dataset):
    """PyTorch Dataset for patient admission data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Ensure features and labels are numpy arrays before converting to tensors
        if not isinstance(features, np.ndarray):
            raise TypeError(
                f"Expected features to be numpy array, got {type(features)}"
            )
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"Expected labels to be numpy array, got {type(labels)}")

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
            # For pipelines containing transformers like OneHotEncoder
            feature_names = pipe.get_feature_names_out(features)
            output_features.extend(feature_names)
        elif hasattr(pipe, "steps"):  # Handle Pipelines
            # Get the last step (usually the transformer like OneHotEncoder or StandardScaler)
            last_step = pipe.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                # Prepend step names if needed, adjust based on pipeline structure
                # This assumes the OneHotEncoder is the last step generating new names
                transformed_names = last_step.get_feature_names_out(features)
                output_features.extend(transformed_names)
            else:
                # For transformers like StandardScaler, names remain the same
                output_features.extend(features)
        else:
            # For simple transformers or if names don't change
            output_features.extend(features)
    return output_features


class SurvivalPredictor(nn.Module):
    """Simple MLP for survival time prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        # TODO: Define model layers (e.g., linear layers, activations)
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


def plot_losses(train_losses: list[float], val_losses: list[float]):
    """Plots training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png")  # Save the plot
    print("Saved training/validation loss plot to training_validation_loss.png")


def _get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer, X_val: pd.DataFrame
) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Handles different sklearn versions and fallback methods.
    """
    try:
        # For sklearn >= 0.24
        return preprocessor.get_feature_names_out()
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
            return []

        return list(numeric_feature_names) + list(cat_feature_names_out)


def _create_descriptive_feature_names(
    encoded_feature_names: list[str], icd9_map: dict
) -> list[str]:
    """Create human-readable feature names from encoded feature names.

    Handles ICD9 codes by adding their descriptions and cleans up prefixes.
    """
    descriptive_names = []
    for name in encoded_feature_names:
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

    return descriptive_names


def calculate_feature_importance(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    X_val: pd.DataFrame,  # Original validation features before preprocessing
    y_val: pd.Series,
    icd9_map: dict,  # Mapping from ICD9 code to description
    n_top_features: int = 20,
    n_repeats: int = 5,
    random_state: int | None = None,
    device: torch.device | str = "cpu",
):
    """Calculates and plots feature importance using permutation importance."""
    print("Calculating feature importance...")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    try:
        # Get feature names from the preprocessor
        encoded_feature_names = _get_feature_names_from_preprocessor(
            preprocessor, X_val
        )

        # Create descriptive names for plotting
        descriptive_feature_names = _create_descriptive_feature_names(
            encoded_feature_names, icd9_map
        )

        # Preprocess the validation data
        X_val_processed = preprocessor.transform(X_val)
    except Exception as e:
        print(
            f"Error during preprocessing or getting feature names in importance calculation: {e}"
        )
        print("Columns available in X_val:", X_val.columns.tolist())
        print("Preprocessor details:", preprocessor)
        return

    # Convert to tensor
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    # Move model to the same device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Create a DataFrame with processed features for easier manipulation
    # Removed unused X_val_processed_df assignment

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
    top_n_indices = np.argsort(importances_np)[-n_top_features:]

    top_n_descriptive_names = [descriptive_feature_names[i] for i in top_n_indices]
    top_n_importances = [importances[i] for i in top_n_indices]

    # Plot Feature Importances
    plt.figure(figsize=(10, n_top_features / 2))  # Adjust height based on N
    plt.barh(range(n_top_features), top_n_importances, align="center")
    plt.yticks(range(n_top_features), top_n_descriptive_names)
    plt.xlabel("Permutation Importance (Increase in MSE)")
    plt.title(f"Top {n_top_features} Feature Importances")
    plt.tight_layout()

    # Save the plot
    out_path = "assets/feature_importance.png"
    plt.savefig(out_path)
    print(f"Feature importance plot saved to {out_path}")

    return top_n_descriptive_names, top_n_importances


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
        "NUM_DIAGNOSES",
        "NUM_PROCEDURES",  # Exclude counts for this run
    ]
    X_original = X_original.drop(
        columns=[col for col in columns_to_drop if col in X_original.columns],
        errors="ignore",
    )

    print(f"Columns for feature importance: {X_original.columns.tolist()}")
    return X_original, y_original


def run_feature_importance_analysis(
    model: nn.Module,
    preprocessor: ColumnTransformer,
    raw_df: pd.DataFrame,
    y_val: pd.Series,
    icd9_map: dict,
    target_column: str = "SURVIVAL_DAYS",
    n_repeats: int = 2,
    n_top_features: int = 20,
    random_state: int = 42,
) -> None:
    """Run feature importance analysis on the model.

    Args:
        model: Trained neural network model
        preprocessor: Fitted preprocessor
        raw_df: Raw dataframe with all columns
        y_val: Validation target values
        icd9_map: Dictionary mapping ICD9 codes to descriptions
        target_column: Name of the target column
        n_repeats: Number of permutation repeats
        n_top_features: Number of top features to display
        random_state: Random seed for reproducibility
    """
    # Prepare data for feature importance
    X_original, y_original = _prepare_data_for_feature_importance(raw_df, target_column)

    # Split data to match the same validation set used in training
    _, X_val_original, _, _ = train_test_split(
        X_original, y_original, test_size=0.2, random_state=random_state
    )

    # Check if X_val_original is available and has columns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_feature_importance(
        model,
        preprocessor=preprocessor,
        X_val=X_val_original,
        y_val=y_val,
        icd9_map=icd9_map,
        n_repeats=n_repeats,
        n_top_features=n_top_features,
        random_state=random_state,
        device=device,
    )


def main():
    """Main execution function."""
    # 1. Load Data: convert tables to dataframes, and merge frames to get the
    # features we're interested in
    raw_df, icd9_map = load_data("../mimic3.db")

    # 2. Preprocess Data (Calculates target, handles features, splits)
    # Use copy to avoid modifying original raw_df
    X_processed, y, preprocessor = preprocess_data(raw_df.copy())

    # Split data *after* preprocessing
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Convert to numpy arrays before creating Tensors (DataFrames might cause issues)
    X_train_np = X_train
    X_val_np = X_val
    y_train_np = y_train.values
    y_val_np = y_val.values

    # 3. Create Datasets and DataLoaders
    train_dataset = PatientDataset(X_train_np, y_train_np)
    val_dataset = PatientDataset(X_val_np, y_val_np)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 4. Initialize Model
    input_dim = X_train_np.shape[1]  # Number of features after preprocessing
    model = SurvivalPredictor(input_dim=input_dim)

    # 5. Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 6. Initial Validation
    val_loss = evaluate_model(model, val_loader)
    print(f"Initial Validation Loss (MSE): {val_loss:.4f}")

    # 7. Train the model
    train_losses = []
    val_losses = []
    for epoch in range(10):
        model, epoch_train_losses, epoch_val_losses = train_model(
            model, train_loader, val_loader, epochs=1, learning_rate=0.001
        )
        train_loss = epoch_train_losses[0]
        val_loss = epoch_val_losses[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    print("Training finished.")
    print(f"Final Validation Loss (MSE): {val_losses[-1]:.4f}")

    # 8. Plot Training/Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), train_losses, label="Training Loss")
    plt.plot(range(1, 11), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    out_path = "assets/training_validation_loss.png"
    plt.savefig(out_path)
    print(f"Saved training/validation loss plot to {out_path}")

    # 9. Feature Importance Analysis
    run_feature_importance_analysis(
        model=model,
        preprocessor=preprocessor,
        raw_df=raw_df.copy(),
        y_val=y_val,
        icd9_map=icd9_map,
        target_column=TARGET_COLUMN,
        n_repeats=2,
        n_top_features=20,
        random_state=42,
    )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
