"""Thin Ultralytics evaluation wrapper."""

from typing import Any


def evaluate_yolo(
    model: Any, data_yaml_path: str, *, imgsz: int = 640, device: str = "cpu"
) -> dict[str, float]:
    """Run validation and return fAIr-friendly metric keys."""
    metrics = model.val(data=data_yaml_path, imgsz=imgsz, device=device)
    # Ultralytics exposes metric values as attributes on `metrics.box`.
    box = metrics.box
    return {
        "accuracy": float(box.map50),
        "mean_iou": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }

