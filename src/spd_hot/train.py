"""Thin Ultralytics training wrapper."""

from pathlib import Path
from typing import Any

from .params import TrainParams


def train_yolo(
    weights_path: str,
    data_yaml_path: str | Path,
    cfg: TrainParams,
    *,
    project_dir: str | Path,
    run_name: str = "spd_hot_train",
) -> Any:
    """Train YOLO and return the trained Ultralytics model object."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    model.train(
        data=str(data_yaml_path),
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        lr0=cfg.learning_rate,
        freeze=10 if cfg.freeze_encoder else 0,
        imgsz=cfg.imgsz,
        device=cfg.device,
        project=str(project_dir),
        name=run_name,
    )
    return model

