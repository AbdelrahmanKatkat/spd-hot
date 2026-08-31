"""Centralised parameters for `spd-hot` (defaults + parsing).

Design goals:
- Single place to define knobs (training / preprocess / inference / postprocess).
- Defaults only; never block STAC from overriding values.
- Accepts plain dicts (from fAIr `pipeline.py`) and returns typed dataclasses.
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SplitParams:
    """Train/val split controls (used by dataset preparation)."""

    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class TrainParams:
    """Ultralytics training parameters (a conservative subset)."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.01
    freeze_encoder: bool = True
    # Optional fractional subsampling for smoke tests and CI.
    sample_fraction: float = 1.0


@dataclass(frozen=True)
class PreprocessParams:
    """Controls how chips are converted to model inputs."""

    # The model’s expected input size (e.g. 640 for YOLO11).
    model_input_size: int = 640
    # Chips coming from fAIr are usually 256×256, but allow any size.
    chip_size_hint: int = 256

    # Normalization: uint8 [0..255] -> float32 [0..1]
    normalize_01: bool = True


@dataclass(frozen=True)
class PostprocessParams:
    """Controls how raw model outputs become final geospatial features."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Filtering in chip pixel space before georeferencing.
    min_box_area_px: float = 0.0


@dataclass(frozen=True)
class InferenceParams:
    """The params surface intended for fAIr inference requests.

    This should remain stable; training-only knobs belong in TrainParams.
    """

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    min_box_area_px: float = 0.0


def _get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d and d[key] is not None else default


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        raw = v.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {v!r}")


def _as_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid int here")
    return int(v)


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid float here")
    return float(v)


def parse_split_params(hyperparameters: Mapping[str, Any] | None) -> SplitParams:
    hp = hyperparameters or {}
    return SplitParams(
        val_ratio=_as_float(_get(hp, "val_ratio", SplitParams.val_ratio)),
        split_seed=_as_int(_get(hp, "split_seed", SplitParams.split_seed)),
        block_size=_as_int(_get(hp, "block_size", SplitParams.block_size)),
    )


def parse_train_params(hyperparameters: Mapping[str, Any] | None) -> TrainParams:
    hp = hyperparameters or {}
    return TrainParams(
        epochs=_as_int(_get(hp, "epochs", TrainParams.epochs)),
        batch_size=_as_int(_get(hp, "batch_size", TrainParams.batch_size)),
        learning_rate=_as_float(_get(hp, "learning_rate", TrainParams.learning_rate)),
        freeze_encoder=_as_bool(_get(hp, "freeze_encoder", TrainParams.freeze_encoder)),
        sample_fraction=_as_float(_get(hp, "sample_fraction", TrainParams.sample_fraction)),
    )


def parse_preprocess_params(hyperparameters: Mapping[str, Any] | None) -> PreprocessParams:
    hp = hyperparameters or {}
    return PreprocessParams(
        model_input_size=_as_int(_get(hp, "model_input_size", PreprocessParams.model_input_size)),
        chip_size_hint=_as_int(_get(hp, "chip_size", PreprocessParams.chip_size_hint)),
        normalize_01=_as_bool(_get(hp, "normalize_01", PreprocessParams.normalize_01)),
    )


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    p = params or {}
    return PostprocessParams(
        confidence_threshold=_as_float(_get(p, "confidence_threshold", PostprocessParams.confidence_threshold)),
        iou_threshold=_as_float(_get(p, "iou_threshold", PostprocessParams.iou_threshold)),
        min_box_area_px=_as_float(_get(p, "min_box_area_px", PostprocessParams.min_box_area_px)),
    )


def parse_inference_params(params: Mapping[str, Any] | None) -> InferenceParams:
    """Parse inference params (what fAIr passes to `predict(..., params=...)`)."""
    post = parse_postprocess_params(params)
    return InferenceParams(
        confidence_threshold=post.confidence_threshold,
        iou_threshold=post.iou_threshold,
        min_box_area_px=post.min_box_area_px,
    )


