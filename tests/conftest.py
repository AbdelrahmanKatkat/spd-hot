import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

CHIP_NAME = "OAM-524288-524288-20.tif"  # a zoom-20 tile at the equator, the usual fAIr chip
CHIP_SIZE = 10  # pixels; a tiny chip keeps the model-space arithmetic readable


def write_chip(path, *, left=0.0, top=CHIP_SIZE, res=1.0, crs="EPSG:4326", bands=3, size=CHIP_SIZE):
    """A georeferenced uint8 GeoTIFF of mid grey; the default is 10 x 10 px, one degree per pixel, from (0, 10)."""
    data = np.full((bands, size, size), 128, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=bands,
        dtype=np.uint8,
        transform=from_origin(left, top, res, res),
        crs=crs,
    ) as dst:
        dst.write(data)
    return path


def write_labels(path, features):
    """A GeoJSON FeatureCollection of `features`, given as `(geometry, properties)` pairs."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props, "geometry": geometry} for geometry, props in features
        ],
    }
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def square(x1, y1, x2, y2):
    """A closed polygon ring as a GeoJSON Polygon geometry."""
    return {"type": "Polygon", "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]}


def yolo_output(boxes, scores, anchors=None):
    """A `(1, 4 + 1, anchors)` YOLO output: `boxes` are `(cx, cy, w, h)` in model pixels, one score each.

    Anchors beyond the given boxes get score 0, so they are never kept.
    """
    # The real output has 8400 anchors; the layout check needs more anchors than features.
    anchors = anchors or max(16, len(boxes))
    out = np.zeros((1, 5, anchors), dtype=np.float32)
    for i, ((cx, cy, w, h), score) in enumerate(zip(boxes, scores)):
        out[0, :, i] = (cx, cy, w, h, score)
    return out


class FakeSession:
    """Stands in for `onnxruntime.InferenceSession`: returns the same output for every chip, and records the inputs."""

    def __init__(self, output):
        self.output = output
        self.inputs = []
        self._input = SimpleNamespace(name="images")

    def get_inputs(self):
        return [self._input]

    def run(self, _outputs, feed):
        (array,) = feed.values()
        self.inputs.append(array)
        return [self.output]


@pytest.fixture
def chips_dir(tmp_path):
    folder = tmp_path / "chips"
    folder.mkdir()
    write_chip(folder / CHIP_NAME)
    return folder
