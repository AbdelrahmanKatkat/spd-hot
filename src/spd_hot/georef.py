"""Geometry conversion helpers (chip pixels -> EPSG:4326 polygons)."""

from typing import Any

from pyproj import Transformer

from .preprocess import PreprocessMeta


def clip_xyxy_to_image(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Clip bbox coordinates to chip boundaries."""
    cx1 = max(0.0, min(float(width), x1))
    cy1 = max(0.0, min(float(height), y1))
    cx2 = max(0.0, min(float(width), x2))
    cy2 = max(0.0, min(float(height), y2))
    return cx1, cy1, cx2, cy2


def area_xyxy(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def unscale_from_model_space(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    meta: PreprocessMeta,
) -> tuple[float, float, float, float]:
    """Map detections from model-input pixels back to original chip pixels."""
    ux1 = x1 / meta.scale_x
    uy1 = y1 / meta.scale_y
    ux2 = x2 / meta.scale_x
    uy2 = y2 / meta.scale_y
    return ux1, uy1, ux2, uy2


def xyxy_to_epsg4326_polygon(x1: float, y1: float, x2: float, y2: float, transform: Any, crs: Any) -> list[list[float]]:
    """Convert one bbox from chip pixels to EPSG:4326 polygon coordinates."""
    corners_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    corners_crs = [transform * (col, row) for col, row in corners_pixel]

    if crs is not None and str(crs) != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return [list(transformer.transform(cx, cy)) for cx, cy in corners_crs]

    return [list(c) for c in corners_crs]

