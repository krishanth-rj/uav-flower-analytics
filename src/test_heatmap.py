"""Run the project inference-to-density-grid integration on one image.

This entry point reuses the project's prediction, spatial-extraction, and
density-grid functions. It prints the resulting 8 by 8 NumPy array and does
not save an annotated image or create a visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    # Support module execution: ``python -m src.test_heatmap ...``.
    from .heatmap import generate_density_grid
    from .predict import load_model, predict_image, validate_image_path
    from .spatial import extract_spatial_data
else:
    # Support direct execution from the repository root: ``python src/test_heatmap.py ...``.
    from heatmap import generate_density_grid
    from predict import load_model, predict_image, validate_image_path
    from spatial import extract_spatial_data


def parse_args() -> argparse.Namespace:
    """Parse the single input image required for the integration run."""
    parser = argparse.ArgumentParser(
        description="Generate and print an 8x8 flower density grid for one image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    return parser.parse_args()


def generate_image_density_grid(image_path: Path) -> np.ndarray:
    """Run inference and return the 8 by 8 density grid for one image.

    Args:
        image_path: Path to an existing input image.

    Returns:
        An 8 by 8 NumPy array containing flower counts per spatial cell.
    """
    model = load_model()
    result = predict_image(model, image_path)
    spatial_data = extract_spatial_data(result)

    # Ultralytics stores original image dimensions as (height, width).
    image_height, image_width = result.orig_shape
    return generate_density_grid(
        spatial_data,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=8,
    )


def main() -> None:
    """Validate the input image, generate its grid, and print the array."""
    args = parse_args()

    try:
        image_path = validate_image_path(args.image)
        density_grid = generate_image_density_grid(image_path)
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(density_grid)


if __name__ == "__main__":
    main()
