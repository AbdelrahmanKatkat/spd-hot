"""Ultralytics fine-tuning of the swimming-pool YOLO11 (needs the `train` extra; import explicitly, not from `spd_hot`).

The base weights come from a checkpoint (`.pt`). With `freeze_encoder` the YOLO11 backbone stays frozen, so the neck and
the head adapt to the new imagery while what the backbone learned is kept. The learning-rate schedule and the
augmentation are Ultralytics' own.
"""

import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .params import FROZEN_LAYERS, PreprocessParams, TrainParams

log = logging.getLogger(__name__)

# Called after each epoch with (epoch starting at 1, train loss, validation loss or None).
EpochCallback = Callable[[int, float, "float | None"], None]


def _ensure_visible_progress() -> None:
    """Make INFO progress lines appear even when the host process never configured logging.

    A bare container step has no root handler and a WARNING root level, so `log.info` would be silent while training
    runs. If nothing is configured, attach a stdout handler to this logger only. A host that configured logging
    itself (ZenML, `basicConfig`) is left alone.
    """
    if not log.isEnabledFor(logging.INFO):
        log.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.propagate = False


def _resolve_device(requested: str) -> str:
    """`auto`, `cuda` or `gpu` pick the first GPU when one is visible; a number picks that GPU; otherwise CPU."""
    import torch

    raw = requested.strip().lower()
    if raw == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if raw in {"auto", "cuda", "gpu"}:
        return "0"
    return raw


def _epoch_losses(trainer: Any) -> tuple[float, float | None]:
    """Train loss: the sum of the epoch's mean loss items. Validation loss: the sum of the `val/*_loss` metrics."""
    tloss = trainer.tloss or {}
    train_loss = sum(float(value) for value in tloss.values()) if tloss else math.nan
    val_losses = [
        float(value)
        for key, value in trainer.metrics.items()
        if key.startswith("val/") and key.endswith("_loss")
    ]
    return train_loss, (sum(val_losses) if val_losses else None)


def _epoch_report(total_epochs: int, on_epoch_end: EpochCallback | None) -> Callable[[Any], None]:
    """The Ultralytics `on_fit_epoch_end` hook: one log line per epoch, then the caller's callback."""

    def report(trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        train_loss, val_loss = _epoch_losses(trainer)
        if not math.isfinite(train_loss):
            log.warning("epoch %d ended with an unavailable or non-finite train loss", epoch)
        log.info(
            "epoch %d/%d train_loss=%.5f val_loss=%s",
            epoch,
            total_epochs,
            train_loss,
            "n/a" if val_loss is None else f"{val_loss:.5f}",
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss)

    return report


def train(
    *,
    weights: str | Path,
    dataset_yaml: str | Path,
    cfg: TrainParams,
    pre: PreprocessParams,
    project_dir: str | Path,
    run_name: str = "swimming_pool",
    on_epoch_end: EpochCallback | None = None,
) -> Any:
    """Fine-tune the checkpoint on the folder of `dataset_yaml`; returns the trained model (its best weights)."""
    from ultralytics import YOLO

    _ensure_visible_progress()
    device = _resolve_device(cfg.device)
    if device == "cpu":
        log.warning("YOLO training is running on CPU; use a visible CUDA GPU for practical fine-tuning speed")
    model = YOLO(str(weights))
    model.add_callback("on_fit_epoch_end", _epoch_report(cfg.epochs, on_epoch_end))
    log.info(
        "training start: weights=%s data=%s epochs=%d batch=%d optimizer=%s lr=%g imgsz=%d device=%s",
        weights,
        dataset_yaml,
        cfg.epochs,
        cfg.batch_size,
        cfg.optimizer,
        cfg.learning_rate,
        pre.model_input_size,
        device,
    )
    model.train(
        data=str(dataset_yaml),
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        optimizer=cfg.optimizer,
        lr0=cfg.learning_rate,
        patience=cfg.patience,
        cos_lr=cfg.cos_lr,
        flipud=cfg.flipud,
        scale=cfg.scale,
        translate=cfg.translate,
        freeze=FROZEN_LAYERS if cfg.freeze_encoder else 0,
        imgsz=pre.model_input_size,
        device=device,
        project=str(project_dir),
        name=run_name,
        verbose=True,
        plots=False,
    )
    return model
