"""Run flower segmentation inference on one image using a trained YOLO model.

This module deliberately contains only the single-image inference workflow.  It
can be imported by other research scripts or executed directly from the
repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "runs/segment/models/training/flower_segmentation/weights/best.pt"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> YOLO:
    """Load the trained Ultralytics segmentation model.

    Args:
        model_path: Path to the model weights file.

    Returns:
        The loaded Ultralytics YOLO model.

    Raises:
        FileNotFoundError: If the requested model weights do not exist.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {model_path}")

    return YOLO(model_path)


def validate_image_path(image_path: Path) -> Path:
    """Validate and normalize an input image path before inference.

    Args:
        image_path: User-supplied path to a single image.

    Returns:
        An absolute path to an existing regular file.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
    """
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Input image is not a file: {image_path}")

    return resolved_path


def predict_image(model: YOLO, image_path: Path) -> Results:
    """Run segmentation inference and return the raw result for one image.

    Args:
        model: A loaded Ultralytics YOLO segmentation model.
        image_path: Path to the image to segment.

    Returns:
        The raw :class:`ultralytics.engine.results.Results` object.

    Raises:
        RuntimeError: If Ultralytics returns no result for the input image.
    """
    # The Python API returns a list, even for exactly one source image.
    predictions = model.predict(source=str(image_path), verbose=False)
    if not predictions:
        raise RuntimeError(f"No prediction result was produced for: {image_path}")

    return predictions[0]


def save_prediction(result: Results, output_path: Path) -> Path:
    """Save the annotated prediction image at the requested output path.

    Args:
        result: Raw Ultralytics result returned by :func:`predict_image`.
        output_path: Destination for the annotated image.

    Returns:
        The path of the saved annotated image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Results.save() renders masks, boxes, labels, and confidence scores.
    result.save(filename=str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-image segmentation inference."""
    parser = argparse.ArgumentParser(
        description="Run flower segmentation inference on a single image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    return parser.parse_args()


def main() -> None:
    """Run prediction from the command line and save the annotated image."""
    args = parse_args()

    try:
        image_path = validate_image_path(args.image)
        model = load_model()
        result = predict_image(model, image_path)
        output_path = save_prediction(result, DEFAULT_RESULTS_DIR / image_path.name)
    except (FileNotFoundError, RuntimeError) as error:
        # argparse prints a concise, conventional command-line error message.
        raise SystemExit(f"Error: {error}") from error

    print(f"Saved annotated prediction to: {output_path}")


if __name__ == "__main__":
    main()
