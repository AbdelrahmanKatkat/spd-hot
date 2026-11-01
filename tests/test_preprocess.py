import numpy as np
import pytest

from spd_hot.params import PreprocessParams
from spd_hot.preprocess import preprocess_chip, preprocess_rgb, read_chip

from .conftest import CHIP_NAME, write_chip


def test_preprocess_rgb_gives_chw_float_in_unit_range():
    rgb = np.full((10, 20, 3), 128, dtype=np.uint8)
    image, meta = preprocess_rgb(rgb, 64)
    assert image.shape == (3, 64, 64)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert (meta.width, meta.height) == (20, 10)
    assert (meta.resized_width, meta.resized_height) == (64, 64)
    assert meta.scale_x == pytest.approx(64 / 20)
    assert meta.scale_y == pytest.approx(64 / 10)


def test_preprocess_chip_reads_the_file_and_keeps_the_georeference(chips_dir):
    image, meta = preprocess_chip(chips_dir / CHIP_NAME, PreprocessParams(model_input_size=32))
    assert image.shape == (3, 32, 32)
    assert meta.crs is not None and str(meta.crs) == "EPSG:4326"
    assert meta.transform.c == 0.0 and meta.transform.f == 10.0


def test_read_chip_refuses_fewer_than_three_bands(tmp_path):
    chip = write_chip(tmp_path / CHIP_NAME, bands=2)
    with pytest.raises(ValueError, match="RGB"):
        read_chip(chip)


def test_preprocess_rgb_refuses_a_grey_array():
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess_rgb(np.zeros((10, 10), dtype=np.uint8), 32)
