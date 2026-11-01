"""OAM chip name -> ground resolution (numpy-free; no CRS, no ONNX layout).

fAIr names every chip `OAM-<x>-<y>-<z>.<ext>` after its Web-Mercator XYZ tile, so the zoom `z` and the tile row `y`
(hence the latitude) are known even for a plain PNG/JPEG without georeferencing.
"""

import math
import re

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)
CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
WEB_MERCATOR_WIDTH_M = 40075016.68558  # width of the zoom-0 tile at the equator


def parse_oam_tile(name: str) -> tuple[int, int, int]:
    """`(x, y, z)` from an OAM chip file name; anything else raises `ValueError`."""
    match = OAM_TILE_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
    x, y, z = (int(match.group(i)) for i in (1, 2, 3))
    if z > 30 or x >= 2**z or y >= 2**z:
        raise ValueError(f"chip name {name!r}: tile ({x}, {y}) is outside the zoom-{z} grid")
    return x, y, z


def tile_gsd_m(name: str, width_px: int) -> float:
    """True ground size of one chip pixel in metres, at the latitude of the tile centre."""
    _, y, z = parse_oam_tile(name)
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / 2**z)))
    return WEB_MERCATOR_WIDTH_M * math.cos(lat) / 2**z / width_px


def zoom_for_gsd_m(
    gsd_m: float,
    lat: float = 0.0,
    width_px: int = 256,
    *,
    min_zoom: int = 18,
    max_zoom: int = 22,
) -> int:
    """Web Mercator zoom whose `width_px` tile GSD is closest to native `gsd_m` at `lat`.

    Clamped to `min_zoom`..`max_zoom` so a typical pool stays more than a few pixels and a
    3 cm drone scene is not written as an empty upsample of a coarser mosaic. fAIr chips may
    mix zooms; `OAM-<x>-<y>-<z>` already encodes `z`.
    """
    if gsd_m <= 0 or width_px <= 0:
        raise ValueError("gsd_m and width_px must be positive")
    cos_lat = math.cos(math.radians(lat))
    if cos_lat <= 1e-12:
        raise ValueError("latitude must be inside the Web Mercator range")
    exact = math.log2(WEB_MERCATOR_WIDTH_M * cos_lat / (gsd_m * width_px))
    return int(max(min_zoom, min(max_zoom, round(exact))))


def tile_footprint_m(name: str, width_px: int, height_px: int | None = None) -> tuple[float, float]:
    """Ground `(width, height)` in metres that the chip covers, at the latitude of the tile centre."""
    height = width_px if height_px is None else height_px
    if width_px <= 0 or height <= 0:
        raise ValueError("tile dimensions must be positive")
    gsd = tile_gsd_m(name, width_px)
    return gsd * width_px, gsd * height
