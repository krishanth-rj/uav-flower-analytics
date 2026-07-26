"""Command-line demonstration of the end-to-end UAV flower analytics workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    # Support module execution: ``python -m src.demo ...``.
    from .analyze import AnalysisValue, analyze_predictions
    from .heatmap import generate_density_grid
    from .predict import (
        DEFAULT_RESULTS_DIR,
        PROJECT_ROOT,
        load_model,
        predict_image,
        save_prediction,
        validate_image_path,
    )
    from .spatial import extract_spatial_data
    from .visualize import plot_density_grid
else:
    # Support direct execution: ``python src/demo.py ...``.
    from analyze import AnalysisValue, analyze_predictions
    from heatmap import generate_density_grid
    from predict import (
        DEFAULT_RESULTS_DIR,
        PROJECT_ROOT,
        load_model,
        predict_image,
        save_prediction,
        validate_image_path,
    )
    from spatial import extract_spatial_data
    from visualize import plot_density_grid


def parse_args() -> argparse.Namespace:
    """Parse the required input image path from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the UAV Flower Analytics workflow on one image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    return parser.parse_args()


def run_demo(image_path: Path) -> tuple[dict[str, AnalysisValue], Path, Path]:
    """Run the complete reusable analytics workflow for one validated image.

    Args:
        image_path: Path to the input image, already validated by
            :func:`validate_image_path`.

    Returns:
        The detection analysis, annotated-image path, and heatmap path.
    """
    model = load_model()
    result = predict_image(model, image_path)

    annotated_path = save_prediction(result, DEFAULT_RESULTS_DIR / image_path.name)
    analysis = analyze_predictions(result)
    spatial_data = extract_spatial_data(result)

    # Ultralytics records image dimensions in (height, width) order.
    image_height, image_width = result.orig_shape
    density_grid = generate_density_grid(
        spatial_data,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_size=8,
    )
    heatmap_path = DEFAULT_RESULTS_DIR / "density_heatmap.png"
    plot_density_grid(density_grid, save_path=heatmap_path, show=False)

    return analysis, annotated_path, heatmap_path


def _display_path(path: Path) -> Path:
    """Return a project-relative path when possible for concise console output."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def print_summary(
    analysis: dict[str, AnalysisValue],
    annotated_path: Path,
    heatmap_path: Path,
) -> None:
    """Print the compact summary for a completed analytics run."""
    print("===== UAV Flower Analytics =====")
    print(f"Flowers detected : {analysis['number_of_flowers']}")
    print(f"Buds detected    : {analysis['number_of_buds']}")
    print(f"Total detections : {analysis['total_detections']}")
    print(f"Image size       : {analysis['image_width']} × {analysis['image_height']}")
    print()
    print(f"Annotated image  : {_display_path(annotated_path)}")
    print(f"Heatmap          : {_display_path(heatmap_path)}")


def main() -> None:
    """Execute the demo from the command line and report its outputs."""
    args = parse_args()
    try:
        image_path = validate_image_path(args.image)
        analysis, annotated_path, heatmap_path = run_demo(image_path)
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print_summary(analysis, annotated_path, heatmap_path)


if __name__ == "__main__":
    main()
