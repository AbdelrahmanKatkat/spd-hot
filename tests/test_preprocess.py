import numpy as np
import rasterio
from rasterio.transform import from_origin

from spd_hot.params import PreprocessParams
from spd_hot.preprocess import preprocess_chip_for_onnx


def test_preprocess_outputs_nchw_float32(tmp_path):
    path = tmp_path / "chip.tif"
    arr = np.full((3, 32, 64), 128, dtype=np.uint8)
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=32,
        count=3,
        dtype=np.uint8,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr)

    batch, meta = preprocess_chip_for_onnx(path, PreprocessParams(model_input_size=64, letterbox=True))
    assert batch.shape == (1, 3, 64, 64)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= 1.0
    assert 0.0 <= float(batch.max()) <= 1.0
    assert meta.width == 64
    assert meta.height == 32

