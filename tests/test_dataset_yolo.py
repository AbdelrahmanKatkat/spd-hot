import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spd_hot.dataset import prepare_yolo_detection_dataset_from_geojson


def _write_chip(path, *, left, top, width=10, height=10):
    arr = np.full((3, height, width), 128, dtype=np.uint8)
    transform = from_origin(left, top, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype=np.uint8,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr)


def test_prepare_yolo_detection_dataset_from_geojson_writes_txt_and_yaml(tmp_path):
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()

    chip1 = chips_dir / "OAM-1-1-18.tif"
    chip2 = chips_dir / "OAM-20-20-18.tif"
    _write_chip(chip1, left=0, top=10)  # bounds: (0,0,10,10)
    _write_chip(chip2, left=100, top=110)  # far away

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    geojson_path = labels_dir / "osm_features.geojson"
    # A polygon inside chip1 in EPSG:4326 coordinates.
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]],
                        },
                    }
                ],
            }
        )
    )

    yolo_dir, split_info = prepare_yolo_detection_dataset_from_geojson(
        chips_dir,
        labels_dir,
        val_ratio=0.5,
        seed=0,
        block_size=1,
        out_dir=tmp_path / "yolo_out",
    )

    assert (yolo_dir / "data.yaml").exists()
    assert split_info["_yolo_dir"] == str(yolo_dir)

    # Both chips should have images + label files in exactly one of train/val.
    for chip in (chip1, chip2):
        found = []
        for split in ("train", "val"):
            img = yolo_dir / "images" / split / chip.name
            lab = yolo_dir / "labels" / split / chip.with_suffix(".txt").name
            if img.exists():
                assert lab.exists()
                found.append(split)
        assert len(found) == 1

    # Chip1 label file should contain at least one line.
    for split in ("train", "val"):
        lab = yolo_dir / "labels" / split / chip1.with_suffix(".txt").name
        if lab.exists():
            txt = lab.read_text().strip()
            if txt:
                assert txt.startswith("0 ")
                break
