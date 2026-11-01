import pytest

from spd_hot.tiles import parse_oam_tile, tile_footprint_m, tile_gsd_m, zoom_for_gsd_m


def test_parse_oam_tile_reads_x_y_z():
    assert parse_oam_tile("OAM-524288-524288-20.tif") == (524288, 524288, 20)
    assert parse_oam_tile("OAM-1-2-5.PNG") == (1, 2, 5)


@pytest.mark.parametrize(
    "name", ["chip.tif", "OAM-1-2.tif", "OAM-1-2-5.gif", "OAM-32-0-5.tif", "OAM-0-0-31.tif"]
)
def test_names_outside_the_oam_grid_are_refused(name):
    with pytest.raises(ValueError):
        parse_oam_tile(name)


def test_zoom_20_equator_chip_is_about_15_cm_per_pixel():
    # Web Mercator at the equator: 40075016.7 m / 2**20 tiles / 256 px = 0.149 m per pixel.
    assert tile_gsd_m("OAM-524288-524288-20.tif", 256) == pytest.approx(0.14929, rel=1e-4)


def test_footprint_is_gsd_times_size():
    width, height = tile_footprint_m("OAM-524288-524288-20.tif", 256, 128)
    gsd = tile_gsd_m("OAM-524288-524288-20.tif", 256)
    assert width == pytest.approx(gsd * 256)
    assert height == pytest.approx(gsd * 128)


@pytest.mark.parametrize(
    ("gsd_m", "expected"),
    [
        (0.14929, 20),
        (0.05, 22),
        (0.08, 21),
        (0.3, 19),
        (0.6, 18),
        (0.01, 22),
        (2.0, 18),
    ],
)
def test_zoom_for_gsd_matches_native_resolution_and_clamps(gsd_m, expected):
    assert zoom_for_gsd_m(gsd_m, lat=0.0) == expected
