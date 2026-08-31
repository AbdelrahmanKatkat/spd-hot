"""Thin Ultralytics evaluation wrapper."""

from typing import Any


def evaluate_yolo(model: Any, data_yaml_path: str) -> dict[str, float]:
    """Run validation and return fAIr-friendly metric keys."""
    metrics = model.val(data=data_yaml_path)
    # Ultralytics exposes metric values as attributes on `metrics.box`.
    box = metrics.box
    return {
        "accuracy": float(box.map50),
        "mean_iou": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }

