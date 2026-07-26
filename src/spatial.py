"""Extract per-instance spatial measurements from Ultralytics predictions.

This module performs no model loading, visualization, command-line parsing, or
file-system access. It only converts an in-memory ``Results`` object into
plain Python spatial records suitable for downstream analysis.
"""

from __future__ import annotations

from ultralytics.engine.results import Results


def _class_name(result: Results, class_id: int) -> str:
    """Resolve a prediction class ID to its human-readable class name."""
    names = result.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _polygon_for_detection(result: Results, index: int) -> list[tuple[float, float]] | None:
    """Return one detection mask polygon in original-image pixel coordinates.

    ``Masks.xy`` is supplied by Ultralytics in pixels relative to the original
    image, unlike ``Masks.xyn``, which is normalized. A result without masks
    is valid (for example, a detection model result), so ``None`` is returned
    when no corresponding polygon is available.
    """
    if result.masks is None or index >= len(result.masks.xy):
        return None

    return [
        (float(point[0]), float(point[1]))
        for point in result.masks.xy[index]
    ]


def extract_spatial_data(result: Results) -> list[dict[str, object]]:
    """Extract object-level spatial data from a single Ultralytics result.

    Bounding-box coordinates are reported in original image pixels. The center
    is represented by ``center_x`` and ``center_y``; ``width`` and ``height``
    are the matching original-pixel box dimensions. Each ``polygon`` is a list
    of ``(x, y)`` points in original-image pixels, or ``None`` if unavailable.

    Args:
        result: One raw ``Results`` object from Ultralytics inference.

    Returns:
        One dictionary per detected object, preserving Ultralytics detection
        order. An empty list is returned when no bounding-box detections exist.
    """
    if result.boxes is None:
        return []

    # ``xywh`` contains centers and dimensions in original-image pixel units.
    boxes_xywh = result.boxes.xywh.tolist()
    class_ids = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    spatial_records: list[dict[str, object]] = []

    for index, (box, class_id, confidence) in enumerate(
        zip(boxes_xywh, class_ids, confidences, strict=True)
    ):
        center_x, center_y, width, height = (float(value) for value in box)
        spatial_records.append(
            {
                "class_name": _class_name(result, int(class_id)),
                "confidence": float(confidence),
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "polygon": _polygon_for_detection(result, index),
            }
        )

    return spatial_records
