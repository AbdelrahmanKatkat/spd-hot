"""Chip -> model input: read the RGB bands, divide by 255 and resize to the model's square input.

The ONNX graph does NOT normalise: its first node is a convolution on `images`, so this module delivers RGB float32
in [0, 1]. That division by 255 is the whole normalisation Ultralytics uses; the model has no mean or std. The base
model saw 512 px images of one zoom-18 tile, so a 256 px zoom-18 chip resized to 512 has the training scale (see
`params.MODEL_INPUT_SIZE`); the resize here does not look at the chip's zoom, so other zooms are not rescaled to it.
The batch axis is added at serving time (`serve.detect_chip`), so one chip is `(3, S, S)` here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .params import PreprocessParams


@dataclass(frozen=True)
class PreprocessMeta:
    """Everything needed to map detections back to chip pixels and then to the map."""

    width: int  # source chip, pixels
    height: int
    resized_width: int  # model input, pixels
    resized_height: int
    transform: Any
    crs: Any

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.height


def read_chip(image_path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Read bands 1-3 as uint8 HWC RGB plus the chip's affine transform and CRS.

    PNG/JPEG chips have no georeferencing: the transform is the identity and the CRS is `None`.
    """
    import rasterio

    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(f"{image_path}: expected >= 3 bands (RGB), found {src.count}")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        return np.clip(rgb, 0, 255).astype(np.uint8), src.transform, src.crs


def preprocess_rgb(
    rgb_hwc: np.ndarray, model_input_size: int, *, transform: Any = None, crs: Any = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """RGB uint8 chip -> `(3, S, S)` float32 in [0, 1] plus `PreprocessMeta`.

    One bilinear resize per channel to `S x S`. The aspect ratio is not kept: OAM chips are square, so nothing is
    distorted.
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb_hwc.shape}")
    h, w, _ = rgb_hwc.shape
    chw = rgb_hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized = np.empty((3, model_input_size, model_input_size), dtype=np.float32)
    for channel in range(3):
        resized[channel] = np.asarray(
            Image.fromarray(chw[channel]).resize(
                (model_input_size, model_input_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    meta = PreprocessMeta(
        width=w,
        height=h,
        resized_width=model_input_size,
        resized_height=model_input_size,
        transform=transform,
        crs=crs,
    )
    return resized, meta


def preprocess_chip(
    image_path: str | Path, cfg: PreprocessParams | None = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """Read a chip and resize it to the model input (`cfg.model_input_size`)."""
    cfg = cfg or PreprocessParams()
    rgb, transform, crs = read_chip(image_path)
    return preprocess_rgb(rgb, cfg.model_input_size, transform=transform, crs=crs)
