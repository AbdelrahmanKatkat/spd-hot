"""Chips + pixel boxes -> the folder Ultralytics trains on (needs the `train` extra; import explicitly).

    <out>/images/{train,val}/OAM-x-y-z.tif    the chips, linked in unchanged (no copy)
    <out>/labels/{train,val}/OAM-x-y-z.txt    one YOLO line per box, normalised to the chip size (empty = no pool)
    <out>/data.yaml                           the paths and the class names

Ultralytics reads each chip at its own size and letterboxes it to the training size, so the labels only need to be
normalised to the chip. The chips must be the files that `build_chip_labels` read.
"""

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rasterio

DEFAULT_CLASS_NAMES = ("swimming_pool",)


def yolo_label_line(box: Mapping[str, Any], width: int, height: int) -> str | None:
    """One chip-pixel XYXY box -> a YOLO `class cx cy w h` line normalised to the chip, or `None` if it is empty."""
    box_w = float(box["x2"]) - float(box["x1"])
    box_h = float(box["y2"]) - float(box["y1"])
    if box_w <= 0 or box_h <= 0:
        return None
    cx = (float(box["x1"]) + box_w / 2) / width
    cy = (float(box["y1"]) + box_h / 2) / height
    return f"{int(box['class_id'])} {cx:.6f} {cy:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def _make_dirs(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty YOLO dataset directory: {out_dir}")
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _link_chip(source: Path, destination: Path) -> None:
    """A YOLO image path without copying the bytes: a symlink, else a hardlink."""
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        try:
            os.link(source.resolve(), destination)
        except OSError as exc:
            raise OSError(f"Could not link {source} into {destination.parent} without copying") from exc


def write_yolo_dataset(
    chips_dir: str | Path,
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    train_names: Sequence[str],
    val_names: Sequence[str],
    out_dir: str | Path,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Path:
    """Write the Ultralytics folder for the named chips; `labels` maps chip name -> pixel boxes. Returns `out_dir`."""
    out = Path(out_dir)
    _make_dirs(out)
    chips = Path(chips_dir)
    for split, names in (("train", train_names), ("val", val_names)):
        for name in names:
            source = chips / name
            _link_chip(source, out / "images" / split / name)
            with rasterio.open(source) as src:
                width, height = src.width, src.height
            lines = [
                line
                for box in labels.get(name, [])
                if (line := yolo_label_line(box, width, height)) is not None
            ]
            (out / "labels" / split / Path(name).with_suffix(".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )
    names_yaml = ", ".join(f"'{name}'" for name in class_names)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(class_names)}\nnames: [{names_yaml}]\n",
        encoding="utf-8",
    )
    return out


def remove_yolo_dataset(out_dir: str | Path) -> None:
    """Delete a folder written by `write_yolo_dataset`, once the run no longer needs it.

    The chips in it are links, so only the links and the small label files are removed: the source chips stay. A folder
    without the `data.yaml` that `write_yolo_dataset` writes is refused, so a wrong path never deletes anything.
    """
    out = Path(out_dir)
    if not (out / "data.yaml").is_file():
        raise ValueError(f"{out} is not a folder written by write_yolo_dataset; nothing was removed")
    shutil.rmtree(out)
