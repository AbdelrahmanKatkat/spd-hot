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

# --- evaluation protocol (Ultralytics' validation, so mAP is comparable with its mAP50 and mAP50-95) ---
MAP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))  # 0.50, 0.55, ..., 0.95
MAP_CONFIDENCE = 0.001  # keep nearly every box, so the precision-recall curve is complete
MAP_NMS_IOU = 0.7  # NMS IoU of the validation


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
    """Ultralytics fine-tuning knobs. The defaults are those of Mozilla's training of the base model
    (`osm-ai-helper`, `demo/finetune_model.ipynb`, with Ultralytics' defaults for what it leaves out), except
    `freeze_encoder`: the YOLO11 backbone (the first 10 layers) stays frozen, so the neck and the head adapt and the
    backbone keeps what it learned instead of forgetting it.

    The optimizer is explicit, AdamW as in Mozilla's training: with Ultralytics' own `auto` choice `learning_rate` is
    ignored and a learning rate is picked from the dataset size. Mozilla's augmentation keeps vertical flips (pools seen
    from above have no up) and turns off scaling and translation.
    """

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "AdamW"
    patience: int = (
        3  # epochs without a better validation fitness before training stops (0 never stops early)
    )
    cos_lr: bool = True  # cosine learning-rate schedule
    flipud: float = 0.5  # probability of a vertical flip
    scale: float = 0.0  # scale augmentation gain
    translate: float = 0.0  # translation augmentation, as a fraction of the image
    freeze_encoder: bool = True
    device: str = (
        "auto"  # a visible CUDA GPU if there is one, else the CPU (training is meant to run on a GPU)
    )


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
    optimizer = str(_get(given, ("optimizer",), TrainParams.optimizer)).strip()
    if not optimizer or optimizer.lower() == "auto":
        raise ValueError(
            f"optimizer must be a named optimizer such as AdamW or SGD, got {optimizer!r}: "
            "'auto' ignores learning_rate"
        )
    patience = _as_int(_get(given, ("patience",), TrainParams.patience))
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    return TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        patience=patience,
        cos_lr=_as_bool(_get(given, ("cos_lr",), TrainParams.cos_lr)),
        flipud=_check_unit("flipud", _as_float(_get(given, ("flipud",), TrainParams.flipud))),
        scale=_check_unit("scale", _as_float(_get(given, ("scale",), TrainParams.scale))),
        translate=_check_unit("translate", _as_float(_get(given, ("translate",), TrainParams.translate))),
        freeze_encoder=_as_bool(
            _get(given, ("freeze_encoder", "freeze_backbone"), TrainParams.freeze_encoder)
        ),
        device=str(_get(given, ("device",), TrainParams.device)),
    )
