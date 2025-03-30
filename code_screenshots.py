"""
Generates syntax-highlighted images of Python functions.

Can be used as a command-line tool or imported.
"""

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

# Use Pygments for highlighting and image generation
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from pygments.styles import get_style_by_name, ClassNotFound

DEFAULT_THEME = "monokai"  # A popular dark theme


def find_function_source(file_path: str, function_name: str) -> str:
    """Finds and returns the source code of a function in a given file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {file_path}: {e}") from e

    function_obj = getattr(module, function_name, None)
    if function_obj is None or not inspect.isfunction(function_obj):
        raise AttributeError(
            f"Error: Function '{function_name}' not found in {file_path}"
        )

    try:
        source_code = inspect.getsource(function_obj)
        # Dedent the source code to remove common leading whitespace
        source_code = inspect.cleandoc(source_code)
        return source_code
    except OSError as e:
        raise OSError(
            f"Error reading source code for '{function_name}' from {file_path}: {e}"
        ) from e


def code_to_image(
    file_path: str,
    function_name: str,
    output_path: str | None = None,
    style_name: str = DEFAULT_THEME,
    font_size: int = 18,  # Increased default font size
    line_numbers: bool = False,  # Added line numbers
    image_format: str = "PNG",
) -> str:
    """
    Generates a syntax-highlighted image for a specific function in a Python file.

    Args:
        file_path: Path to the Python file.
        function_name: Name of the function within the file.
        output_path: Optional path to save the image. If None, defaults to
                     '<function_name>.png' in the current directory.
        style_name: Name of the Pygments style (theme).
        font_size: Font size for the code in the image.
        line_numbers: Whether to include line numbers in the image.
        image_format: The format for the output image (e.g., "PNG", "JPEG").

    Returns:
        The path where the image was saved.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        AttributeError: If the function is not found in the file.
        OSError: If there's an error reading the source code.
        ImportError: If the module cannot be loaded or executed.
        ClassNotFound: If the specified Pygments style is invalid.
        Exception: For errors during image generation.
    """
    print(f"Processing function '{function_name}' from file '{file_path}'...")

    # 1. Get the function source code
    source_code = find_function_source(file_path, function_name)
    print(f"Successfully retrieved source code for '{function_name}'.")

    # 2. Determine output path and ensure directory exists
    if output_path is None:
        # Ensure the output filename has the correct extension
        base_name = f"{function_name}.{image_format.lower()}"
        output_path = base_name
    else:
        # Ensure provided path has the correct extension
        if not output_path.lower().endswith(image_format.lower()):
            output_path = f"{os.path.splitext(output_path)[0]}.{image_format.lower()}"

    output_path = os.path.abspath(output_path)  # Ensure absolute path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating image with style '{style_name}'...")

    # 3. Setup Pygments
    try:
        style = get_style_by_name(style_name)
    except ClassNotFound:
        raise ClassNotFound(
            f"Error: Pygments style '{style_name}' not found. "
            f"Choose from available styles (e.g., monokai, dracula, default)."
        )

    lexer = PythonLexer()
    formatter = ImageFormatter(
        style=style,
        font_size=font_size,
        line_numbers=line_numbers,
        image_format=image_format,
        # line_pad = 10, # Adjust padding if needed
        # font_name = 'monospace' # Specify font if desired
    )

    # 4. Generate and save the image
    try:
        with open(output_path, "wb") as f:
            highlight(source_code, lexer, formatter, outfile=f)
        print(f"Image successfully saved to: {output_path}")
        return output_path
    except Exception as e:
        raise Exception(f"Error generating image using Pygments: {e}") from e


# Excerpt code from a given Jupyter notebook cell into an image suitable for
# use in a slide deck.
# NOTE: Currently unused; storing this function here from a notebook
# where it was in use. There's probably more work to be done to extract code
# from a notebook that's not currently importing this function
def jupyter_code_to_image(
    filename: str,
    code_text: str | None = None,
    style: str = DEFAULT_THEME,
    font_size=12,
    dpi: int = 150,
    format="png",
    line_numbers=False,
):
    """
    Converts Jupyter notebook code cell content to an image.

    Parameters:
    -----------
    code_text: str|None,
        text to be highlighted. If empty, use the currently executing cell
    style : str, default='monokai'
        Syntax highlighting style ('monokai', 'default', 'solarized-dark',
        'solarized-light', etc.)
    font_size : int, default=10
        Font size for the code text.
    dpi : int, default=150
        The resolution of the image in dots per inch.
    format : str, default='png'
        The image format ('png', 'jpg', 'svg', 'pdf', etc.).
    filename : str, default=None
        If provided, the image will be saved to this file.
    figsize : tuple, default=None
        Figure size in inches (width, height). If None, it's calculated based on content.
    line_numbers : bool, default=True
        Whether to include line numbers in the code image.

    Returns:
    --------
    IPython.display.Image or None
        The image object if filename is None, otherwise None.

    Example:
    --------
    # Capture the current cell
    code_to_image()

    # Capture a specific cell (e.g., the 3rd cell, index 2)
    code_to_image(cell_index=2)

    # Save to file with custom styling
    code_to_image(style='solarized-light', filename='my_code.png', dpi=300)
    """

    # In[-1] (in a Jupyter notebook) yields whichever cell was most recently
    # invoked, the calling cell.
    code_text = code_text or In[-1]  # noqa

    # Create an image of the code with syntax highlighting
    formatter = ImageFormatter(
        style=style, font_size=font_size, line_numbers=line_numbers
    )
    image_bytes = highlight(code_text, PythonLexer(), formatter)

    # Save to file
    with open(filename, "wb") as f:
        f.write(image_bytes)
    print(f"Code image saved to {filename}")
    return filename

    # display_image = False
    # if display_image:
    #     from IPython.display import display, Image
    #     img = Image(data=image_bytes)
    #     display(img)
    #     return img


def main():
    """Command-line interface handler."""
    parser = argparse.ArgumentParser(
        description="Generate a syntax-highlighted image of a Python function."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the Python file containing the function.",
    )
    parser.add_argument(
        "-n", "--name", required=True, help="Name of the function to image."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path for the output image file (e.g., my_func.png). "
        "Defaults to <function_name>.png in the current directory.",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--style",
        default=DEFAULT_THEME,
        help=f"Pygments style (theme) name. Default: {DEFAULT_THEME}",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=18,
        help="Font size for the code. Default: 18",
    )
    parser.add_argument(
        "--no-line-numbers",
        action="store_false",  # Sets line_numbers to False if flag is present
        dest="line_numbers",
        help="Omit line numbers from the image.",
    )
    parser.add_argument(
        "--format",
        default="PNG",
        choices=["PNG", "JPEG", "GIF", "BMP", "TIFF"],
        help="Output image format. Default: PNG",
    )

    args = parser.parse_args()

    try:
        saved_path = code_to_image(
            file_path=args.file,
            function_name=args.name,
            output_path=args.output,
            style_name=args.style,
            font_size=args.font_size,
            line_numbers=args.line_numbers,
            image_format=args.format.upper(),
        )
        print(f"\nProcess complete. Image saved at: {saved_path}")
    except (
        FileNotFoundError,
        AttributeError,
        OSError,
        ImportError,
        ClassNotFound,
        Exception,
    ) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
