"""Training-time helpers for converting GeoJSON polygons to YOLO bbox labels.

This is intentionally separate from `georef.py`:
- `georef.py` is inference-oriented (model outputs -> EPSG:4326 polygons).
- this module is training-oriented (label polygons -> YOLO txt bboxes).
"""

from typing import Any


def world_bounds_to_pixel_bbox_xyxy(
    *,
    transform: Any,
    width: int,
    height: int,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> tuple[int, int, int, int]:
    """Convert world-coordinate bounds to a clipped pixel bbox (x1,y1,x2,y2).

    Assumes bounds are already in the chip's CRS.
    """
    from rasterio.transform import rowcol

    row_min, col_min = rowcol(transform, minx, maxy)
    row_max, col_max = rowcol(transform, maxx, miny)

    x1 = max(0, int(col_min))
    y1 = max(0, int(row_min))
    x2 = min(width, int(col_max))
    y2 = min(height, int(row_max))
    return x1, y1, x2, y2


def pixel_bbox_xyxy_to_yolo_txt_line(
    *,
    class_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> str | None:
    """Convert a pixel bbox into one YOLO `.txt` line (normalized 0..1)."""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0 or width <= 0 or height <= 0:
        return None
    cx = (x1 + bw / 2) / width
    cy = (y1 + bh / 2) / height
    nw = bw / width
    nh = bh / height
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

