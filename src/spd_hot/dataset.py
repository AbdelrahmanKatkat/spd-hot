"""Dataset helpers for spd-hot (numpy, rasterio, pyproj; no torch and no shapely, safe at serving time).

- `list_chip_paths`, `spatial_split`: find the chips and split them into train and validation.
- `build_chip_labels`: a GeoJSON of swimming-pool polygons -> the pixel boxes of each chip, nothing copied.

The rules that pick the pools from a source file (tags, indoor pools) are dataset-specific and live in `research/`.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .tiles import CHIP_PATTERNS, parse_oam_tile

LABEL_SUFFIXES = (".geojson", ".json")
POLYGON_TYPES = ("Polygon", "MultiPolygon")


def list_chip_paths(root: str | Path) -> list[Path]:
    """Chips in a folder, sorted by name. Each name must be a canonical OAM name."""
    folder = Path(root)
    paths = sorted(path for pattern in CHIP_PATTERNS for path in folder.glob(pattern) if path.is_file())
    for path in paths:
        parse_oam_tile(path.name)
    return paths


def spatial_split(
    chip_names: Sequence[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; whole blocks of `block_size` tiles go to one side.

    Neighbouring chips share pools cut at chip borders and similar surroundings, so a random chip split leaks.
    Coordinates of different zooms are mapped to the finest zoom first. Names must be `OAM-x-y-z.ext`.
    Returns sorted `(train, val)` name lists.
    """
    parsed = [(name, *parse_oam_tile(name)) for name in chip_names]
    max_zoom = max((zoom for _, _, _, zoom in parsed), default=0)
    blocks_map: dict[tuple[int, int], list[str]] = {}
    for name, tile_x, tile_y, zoom in parsed:
        factor = 2 ** (max_zoom - zoom)
        blocks_map.setdefault((tile_x * factor // block_size, tile_y * factor // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(blocks_map)
    rng.shuffle(blocks)

    n_val_target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0
    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(blocks_map[block])
    return sorted(train), sorted(val)


def _find_labels_geojson(labels: str | Path) -> Path:
    """The labels file: the path itself if it is a `.geojson` / `.json` file, else the one such file under a folder."""
    root = Path(labels)
    if root.is_file() and root.suffix.lower() in LABEL_SUFFIXES:
        return root
    matches = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson (or .json) in {root}, found {len(matches)}")
    return matches[0]


def _parse_geojson_crs(feature_collection: dict[str, Any]) -> str:
    """CRS name from the legacy GeoJSON `crs` member; RFC 7946 files default to EPSG:4326."""
    crs = feature_collection.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties")
        if isinstance(props, dict) and isinstance(props.get("name"), str) and props["name"]:
            return props["name"]
    return "EPSG:4326"


def _iter_xy_coords(node: Any) -> list[tuple[float, float]]:
    """All `(x, y)` pairs in a GeoJSON `coordinates` structure (any nesting, extra dimensions ignored)."""
    pairs: list[tuple[float, float]] = []

    def _walk(item: Any) -> None:
        if not isinstance(item, (list, tuple)):
            return
        if len(item) >= 2 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in item
        ):
            pairs.append((float(item[0]), float(item[1])))
            return
        for child in item:
            _walk(child)

    _walk(node)
    return pairs


def read_label_bounds(labels_geojson: str | Path) -> tuple[np.ndarray, str]:
    """Bounds `(N, 4)` of every polygon as `(minx, miny, maxx, maxy)` in the file CRS, plus that CRS.

    Only polygons are pools. Points (a pool mapped as a node has no outline) and lines are skipped.
    """
    collection = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    features = collection.get("features", [])
    if not isinstance(features, list):
        raise TypeError("GeoJSON FeatureCollection 'features' must be a list")
    bounds: list[tuple[float, float, float, float]] = []
    for feature in features:
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geom, dict) or geom.get("type") not in POLYGON_TYPES:
            continue
        coords = _iter_xy_coords(geom.get("coordinates"))
        if coords:
            xs, ys = zip(*coords)
            bounds.append((min(xs), min(ys), max(xs), max(ys)))
    return np.asarray(bounds, dtype=np.float64).reshape(-1, 4), _parse_geojson_crs(collection)


def _project_bounds(bounds: np.ndarray, src_crs: Any, dst_crs: Any) -> np.ndarray:
    """Reproject `(N, 4)` bounds by projecting all four corners and re-bounding."""
    if len(bounds) == 0 or src_crs is None or dst_crs is None or str(src_crs) == str(dst_crs):
        return bounds
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = np.stack([bounds[:, 0], bounds[:, 0], bounds[:, 2], bounds[:, 2]], axis=1)
    ys = np.stack([bounds[:, 1], bounds[:, 3], bounds[:, 1], bounds[:, 3]], axis=1)
    out_x, out_y = transformer.transform(xs, ys)
    return np.stack([out_x.min(axis=1), out_y.min(axis=1), out_x.max(axis=1), out_y.max(axis=1)], axis=1)


def chip_boxes(
    chip_path: str | Path,
    *,
    label_bounds: np.ndarray,
    labels_crs: Any,
    class_id: int = 0,
    min_box_px: float = 0.0,
) -> list[dict[str, float | int]]:
    """Label bounds -> XYXY boxes in chip pixels, clipped to the chip.

    A box is the bounds of its polygon, clipped to the chip: a pool that crosses the chip edge keeps its part inside.
    Boxes off the chip are dropped, and so are boxes thinner than `min_box_px` after clipping.
    """
    import rasterio

    with rasterio.open(chip_path) as src:
        width, height, transform, crs = int(src.width), int(src.height), src.transform, src.crs
    if crs is None:
        raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced GeoTIFF chip")
    if len(label_bounds) == 0:
        return []

    projected = _project_bounds(label_bounds, labels_crs, crs)
    corner_x = np.stack([projected[:, 0], projected[:, 0], projected[:, 2], projected[:, 2]], axis=1)
    corner_y = np.stack([projected[:, 1], projected[:, 3], projected[:, 1], projected[:, 3]], axis=1)
    inv = ~transform
    # Explicit affine maths (same reason as georef.py): works across `affine` versions.
    cols = inv.a * corner_x + inv.b * corner_y + inv.c
    rows = inv.d * corner_x + inv.e * corner_y + inv.f
    x1, x2 = np.clip(cols.min(axis=1), 0, width), np.clip(cols.max(axis=1), 0, width)
    y1, y2 = np.clip(rows.min(axis=1), 0, height), np.clip(rows.max(axis=1), 0, height)
    keep = (x2 > x1) & (y2 > y1) & (x2 - x1 >= min_box_px) & (y2 - y1 >= min_box_px)
    return [
        {
            "x1": float(left),
            "y1": float(top),
            "x2": float(right),
            "y2": float(bottom),
            "class_id": int(class_id),
        }
        for left, top, right, bottom in zip(x1[keep], y1[keep], x2[keep], y2[keep])
    ]


def build_chip_labels(
    chips_dir: str | Path, labels: str | Path, *, class_id: int = 0, min_box_px: float = 2.0
) -> dict[str, list[dict[str, float | int]]]:
    """Pixel boxes of every chip in `chips_dir` from one GeoJSON of pool polygons.

    `labels` is the `.geojson` / `.json` file, or a folder holding exactly one. Nothing is copied, so a pipeline step can
    call this on the chips and labels it was given. Chips without a pool get an empty list (negative examples).
    """
    bounds, crs = read_label_bounds(_find_labels_geojson(labels))
    return {
        path.name: chip_boxes(
            path, label_bounds=bounds, labels_crs=crs, class_id=class_id, min_box_px=min_box_px
        )
        for path in list_chip_paths(chips_dir)
    }
