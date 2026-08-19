"""Image preprocessing helpers for ONNX inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from .params import PreprocessParams


@dataclass(frozen=True)
class PreprocessMeta:
    """Metadata required to map detections back to source chip coordinates."""

    width: int
    height: int
    scale_x: float
    scale_y: float
    transform: Any
    crs: Any


def _resize_rgb(rgb_hwc: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float]:
    h, w, _ = rgb_hwc.shape
    resized = np.asarray(Image.fromarray(rgb_hwc).resize((target_size, target_size), Image.Resampling.BILINEAR))
    scale_x = target_size / w
    scale_y = target_size / h
    return resized, scale_x, scale_y


def preprocess_chip_for_onnx(image_path: str | Path, cfg: PreprocessParams | None = None) -> tuple[np.ndarray, PreprocessMeta]:
    """Read one chip and return ONNX-ready tensor and metadata.

    Output tensor shape is `(1, 3, size, size)` when `cfg.nchw` is True.
    """

    cfg = cfg or PreprocessParams()

    with rasterio.open(image_path) as src:
        # Read once and convert to HWC (uint8) for one-pass resize.
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        rgb_hwc = np.clip(rgb, 0, 255).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    resized, scale_x, scale_y = _resize_rgb(
        rgb_hwc,
        target_size=cfg.model_input_size,
    )

    arr = resized.astype(np.float32)
    if cfg.normalize_01:
        arr = arr / 255.0

    if cfg.nchw:
        batch = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    else:
        batch = arr[np.newaxis, ...]

    meta = PreprocessMeta(
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
        transform=transform,
        crs=crs,
    )
    return batch.astype(np.float32), meta

