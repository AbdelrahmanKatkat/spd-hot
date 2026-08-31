"""Dataset helpers for YOLO fine-tuning (labels + splitting).

fAIr provides training assets as:
- `dataset_chips`: directory of RGB GeoTIFF chips (often `OAM-x-y-z.tif`)
- `dataset_labels`: directory containing GeoJSON polygon labels (OSM-derived)

Ultralytics YOLO training expects a YOLO dataset directory:
- images/{train,val}/*.tif
- labels/{train,val}/*.txt   (YOLO bbox format, normalized 0..1)
- data.yaml
"""

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def spatial_split(
    chip_names: list[str],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; entire `(x//K, y//K)` blocks pick a side.

    Chips that do not match `OAM-x-y-z` are shuffled and used only to fill leftover val slots.
    """
    matched: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        oam_match = OAM_TILE_RE.match(name)
        if oam_match is None:
            unmatched.append(name)
            continue
        tile_x, tile_y = int(oam_match.group(1)), int(oam_match.group(2))
        matched.setdefault((tile_x // block_size, tile_y // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(matched.keys())
    rng.shuffle(blocks)

    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio)) if n_total else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        bucket = val if len(val) < n_val_target else train
        bucket.extend(matched[block])

    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        remaining = max(0, n_val_target - len(val))
        val.extend(leftover[:remaining])
        train.extend(leftover[remaining:])

    return sorted(train), sorted(val)


def _resolve_geojson(labels_path: str | Path) -> Path:
    p = Path(labels_path)
    if p.is_file():
        if p.suffix.lower() != ".geojson":
            raise ValueError(f"labels_path must be a .geojson file or a directory containing one, got {p}")
        return p
    geojsons = sorted(p.rglob("*.geojson"))
    if not geojsons:
        raise FileNotFoundError(f"No .geojson labels found under {p}")
    if len(geojsons) != 1:
        raise ValueError(f"Expected exactly one .geojson under {p}, found {len(geojsons)}")
    return geojsons[0]


def _labels_crs_from_geojson(data: dict[str, Any]) -> str:
    """Best-effort CRS detection for fAIr label GeoJSON.

    - If a `crs` member exists, we try to read an EPSG-like name from it.
    """
    crs_obj = data.get("crs")
    if not isinstance(crs_obj, dict):
        return "EPSG:4326"
    props = crs_obj.get("properties")
    if not isinstance(props, dict):
        return "EPSG:4326"
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "EPSG:4326"


def _reset_yolo_dirs(yolo_dir: Path) -> None:
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_chip_with_sidecars(src: Path, dst: Path) -> None:
    """Materialize a chip into the YOLO dataset tree.

    Prefer symlinks to avoid duplicating large chip files inside training jobs.
    Fall back to full copies when symlinks are not supported.
    """
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

    # OAM chips sometimes store geotransform / projection as sidecars.
    for ext in (".aux.xml", ".tfw", ".prj"):
        sidecar = src.parent / (src.name + ext)
        if not sidecar.exists():
            continue
        out_sidecar = dst.parent / sidecar.name
        try:
            out_sidecar.symlink_to(sidecar)
        except OSError:
            shutil.copy2(sidecar, out_sidecar)


def prepare_yolo_detection_dataset_from_geojson(
    chips_path: str | Path,
    labels_path: str | Path,
    *,
    class_name: str = "swimming_pool",
    val_ratio: float = 0.2,
    seed: int = 42,
    block_size: int = 4,
    out_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a YOLO detection dataset dir from GeoTIFF chips + GeoJSON polygons.

    Returns `(yolo_dir, split_info)` where `split_info["_yolo_dir"]` points to `yolo_dir`.
    """
    import rasterio
    from .geo_to_yolo_converter import pixel_bbox_xyxy_to_yolo_txt_line, world_bounds_to_pixel_bbox_xyxy

    from shapely.geometry import box, shape
    from shapely.ops import transform as shapely_transform
    from shapely.strtree import STRtree
    from pyproj import Transformer

    chips_root = Path(chips_path)
    if chips_root.is_file():
        raise ValueError(f"chips_path must be a directory, got file: {chips_root}")

    chip_files = sorted([*chips_root.rglob("*.tif"), *chips_root.rglob("*.tiff")])
    if not chip_files:
        raise FileNotFoundError(f"No .tif/.tiff chips found under {chips_root}")

    geojson_path = _resolve_geojson(labels_path)
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    raw_polys = [shape(f["geometry"]) for f in features if isinstance(f, dict) and f.get("geometry")]
    labels_crs = _labels_crs_from_geojson(data)

    # Project polygons once to the chip CRS if needed.
    with rasterio.open(chip_files[0]) as src0:
        chip_crs = src0.crs
    polys = raw_polys
    if chip_crs is not None and str(chip_crs) != labels_crs:
        to_chip = Transformer.from_crs(labels_crs, chip_crs, always_xy=True)
        polys = [
            shapely_transform(lambda x, y, _z=None: to_chip.transform(x, y), p)  # type: ignore[misc]
            for p in raw_polys
        ]

    tree = STRtree(polys) if polys else None

    # Split by chip *names* (stable + consistent with fAIr examples).
    chip_names = [p.name for p in chip_files]
    train_names, val_names = spatial_split(chip_names, val_ratio, seed, block_size=block_size)
    split_of: dict[str, str] = {n: "train" for n in train_names}
    split_of.update({n: "val" for n in val_names})

    # No deterministic cache by default: this is a training-job artifact, and it's safer to
    # build a fresh directory per run unless the caller explicitly chooses a location.
    yolo_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="spd_hot_yolo_dataset_"))
    _reset_yolo_dirs(yolo_dir)

    # Build labels per chip.
    for chip in chip_files:
        split = split_of.get(chip.name, "train")
        out_img = yolo_dir / "images" / split / chip.name
        _copy_chip_with_sidecars(chip, out_img)

        with rasterio.open(chip) as src:
            chip_box = box(*src.bounds)
            transform = src.transform
            width, height = src.width, src.height

        if tree is not None:
            # Shapely 2.x STRtree.query returns candidate indices, not geometries.
            candidate_indices = tree.query(chip_box)
            candidates = [polys[int(i)] for i in np.atleast_1d(candidate_indices)]
        else:
            candidates = polys
        lines: list[str] = []
        for poly in candidates:
            if not poly.intersects(chip_box):
                continue
            clipped = poly.intersection(chip_box)
            if clipped.is_empty:
                continue
            minx, miny, maxx, maxy = clipped.bounds
            x1, y1, x2, y2 = world_bounds_to_pixel_bbox_xyxy(
                transform=transform,
                width=width,
                height=height,
                minx=float(minx),
                miny=float(miny),
                maxx=float(maxx),
                maxy=float(maxy),
            )
            line = pixel_bbox_xyxy_to_yolo_txt_line(
                class_id=0, x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height
            )
            if line is not None:
                lines.append(line)

        label_file = yolo_dir / "labels" / split / chip.with_suffix(".txt").name
        label_file.write_text("\n".join(lines), encoding="utf-8")

    (yolo_dir / "data.yaml").write_text(
        f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['{class_name}']\n",
        encoding="utf-8",
    )

    split_info = {
        "strategy": "spatial",
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "block_size": int(block_size),
        "train_count": len(train_names),
        "val_count": len(val_names),
        "class_name": class_name,
        "_yolo_dir": str(yolo_dir),
    }
    return yolo_dir, split_info
