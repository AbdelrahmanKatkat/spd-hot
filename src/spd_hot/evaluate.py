"""Thin Ultralytics evaluation wrapper."""

from typing import Any


def evaluate_yolo(model: Any, data_yaml_path: str) -> dict[str, float]:
    """Run validation and return fAIr-friendly metric keys."""
    metrics = model.val(data=data_yaml_path)
    # Ultralytics exposes metric values as attributes on `metrics.box`.
    box = metrics.box
    return {
        "fair:map50": float(box.map50),
        "fair:map50_95": float(box.map),
        "fair:precision": float(box.mp),
        "fair:recall": float(box.mr),
    }

