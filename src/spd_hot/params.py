"""Constants, defaults and dict -> dataclass parsing for `spd-hot`.

Parsers tolerate `None` and missing keys. The first present key wins: the canonical name, then its aliases.

Where each value lives (the level-2 ONNX contract):
- inside the ONNX graph: the backbone, neck and head, anchors and box decoding. Fixed by the exported model: input
  `images` (1, 3, 512, 512) float32, output `output0` (1, 4 + nc, anchors). There is NO normalisation in the graph (its
  first node is a convolution, checked on the exported file): the input must already be RGB in [0, 1]. Ultralytics
  applies no mean or std, only the division by 255.
- outside, tunable: the chip resize and the division by 255 (`preprocess.py`, `PreprocessParams`), the confidence
  filter and NMS (`PostprocessParams`).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# --- model constants (fixed by the exported graph) ---
# The base model (mozilla-ai/swimming-pool-detector) was trained at `imgsz=512` on 512 x 512 px images, each one Mapbox
# satellite tile at zoom 18: 40075016.686 * cos(lat) / 2**18 / 512 = 0.2986 * cos(lat) m per pixel. A 256 px chip of the
# same zoom-18 tile (0.597 * cos(lat) m per pixel) resized to 512 therefore has the training scale; other zooms do not.
MODEL_INPUT_SIZE = (
    512  # the ONNX input is fixed at 512 x 512; a size must be a multiple of the YOLO stride (32)
)
MAX_DETECTIONS = 300  # per chip after NMS (Ultralytics' `max_det` default)
FROZEN_LAYERS = 10  # Ultralytics `freeze` when the encoder is frozen: the YOLO11 backbone, layers 0 to 9


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class PreprocessParams:
    """How a chip is resized before the model. Training and serving must use the same value."""

    model_input_size: int = MODEL_INPUT_SIZE


@dataclass(frozen=True)
class PostprocessParams:
    """Thresholds applied outside the ONNX graph (which returns every anchor)."""

    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45  # Ultralytics' `iou`, per class, within a chip


@dataclass(frozen=True)
class TrainParams:
    """Ultralytics fine-tuning knobs. `freeze_encoder` freezes the YOLO11 backbone, so the neck and the head adapt."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.01
    freeze_encoder: bool = True
    device: str = "auto"  # cuda if visible, else cpu (training is meant to run on a GPU)


def _get(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """First present non-None key wins (canonical key first, then aliases)."""
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid int here")
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid float here")
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {value!r}")


def _check_unit(name: str, value: float, *, low_open: bool = False) -> float:
    ok = (0.0 < value <= 1.0) if low_open else (0.0 <= value <= 1.0)
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1]' if low_open else '[0, 1]'}, got {value}")
    return value


def parse_split_params(hyperparameters: Mapping[str, Any] | None) -> SplitParams:
    given = hyperparameters or {}
    val_ratio = _as_float(_get(given, ("val_ratio",), SplitParams.val_ratio))
    block_size = _as_int(_get(given, ("block_size",), SplitParams.block_size))
    if not 0 < val_ratio < 1:  # both sides need chips
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return SplitParams(
        val_ratio=val_ratio,
        split_seed=_as_int(_get(given, ("split_seed",), SplitParams.split_seed)),
        block_size=block_size,
    )


def parse_preprocess_params(params: Mapping[str, Any] | None) -> PreprocessParams:
    """`model_input_size` is canonical; `chip_size` (the fAIr name) is an alias."""
    given = params or {}
    size = _as_int(_get(given, ("model_input_size", "chip_size"), PreprocessParams.model_input_size))
    if size < 32 or size % 32:
        raise ValueError(f"model_input_size must be a multiple of 32 and at least 32, got {size}")
    return PreprocessParams(model_input_size=size)


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    """`confidence_threshold` and `nms_threshold` are canonical; the other names are aliases."""
    given = params or {}
    return PostprocessParams(
        confidence_threshold=_check_unit(
            "confidence_threshold",
            _as_float(
                _get(
                    given,
                    ("confidence_threshold", "score_thresh"),
                    PostprocessParams.confidence_threshold,
                )
            ),
        ),
        nms_threshold=_check_unit(
            "nms_threshold",
            _as_float(
                _get(
                    given,
                    ("nms_threshold", "iou_threshold", "nms_thresh"),
                    PostprocessParams.nms_threshold,
                )
            ),
        ),
    )


def parse_train_params(params: Mapping[str, Any] | None) -> TrainParams:
    """`learning_rate` is canonical, `lr` an alias; `freeze_backbone` is an alias of `freeze_encoder`."""
    given = params or {}
    epochs = _as_int(_get(given, ("epochs",), TrainParams.epochs))
    batch_size = _as_int(_get(given, ("batch_size",), TrainParams.batch_size))
    learning_rate = _as_float(_get(given, ("learning_rate", "lr"), TrainParams.learning_rate))
    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch_size must be >= 1, got {epochs} and {batch_size}")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be a positive number, got {learning_rate}")
    return TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_encoder=_as_bool(
            _get(given, ("freeze_encoder", "freeze_backbone"), TrainParams.freeze_encoder)
        ),
        device=str(_get(given, ("device",), TrainParams.device)),
    )
