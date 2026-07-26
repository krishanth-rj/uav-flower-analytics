"""Extract non-visual summary statistics from a YOLO segmentation result.

The functions in this module only inspect an in-memory Ultralytics ``Results``
object. They do not load models, read or write files, or generate plots.
"""

from __future__ import annotations

from typing import TypeAlias

from ultralytics.engine.results import Results


AnalysisValue: TypeAlias = int | float | list[float]


def _class_name(result: Results, class_id: int) -> str:
    """Return the class name associated with a numeric prediction class ID."""
    names = result.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _average_confidence(confidences: list[float]) -> float:
    """Calculate a class confidence mean, returning zero when it is absent."""
    return sum(confidences) / len(confidences) if confidences else 0.0


def analyze_predictions(result: Results) -> dict[str, AnalysisValue]:
    """Summarize bud and flower detections from one Ultralytics result.

    Class membership is resolved from ``result.names`` rather than assuming a
    fixed class-ID order. Predictions belonging to classes other than ``bud``
    and ``flower`` remain part of ``total_detections`` but are not included in
    either class-specific summary.

    Args:
        result: A single raw ``Results`` object from Ultralytics inference.

    Returns:
        A dictionary containing detection counts, per-class confidence lists
        and averages, and the original image dimensions. A class with no
        detections has an empty confidence list and an average confidence of
        ``0.0``.
    """
    # Ultralytics stores image dimensions as (height, width).
    image_height, image_width = (int(value) for value in result.orig_shape)
    bud_confidences: list[float] = []
    flower_confidences: list[float] = []

    # A valid Results object may contain no detected instances, in which case
    # ``boxes`` is None or has zero rows.
    if result.boxes is not None:
        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        for class_id, confidence in zip(class_ids, confidences, strict=True):
            class_name = _class_name(result, int(class_id)).lower()
            if class_name == "bud":
                bud_confidences.append(float(confidence))
            elif class_name == "flower":
                flower_confidences.append(float(confidence))

    total_detections = len(result.boxes) if result.boxes is not None else 0
    return {
        "total_detections": total_detections,
        "number_of_buds": len(bud_confidences),
        "number_of_flowers": len(flower_confidences),
        "average_confidence_buds": _average_confidence(bud_confidences),
        "average_confidence_flowers": _average_confidence(flower_confidences),
        "bud_confidences": bud_confidences,
        "flower_confidences": flower_confidences,
        "image_width": image_width,
        "image_height": image_height,
    }
