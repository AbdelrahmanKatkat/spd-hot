import pytest

from spd_hot.dataset import build_chip_labels, list_chip_paths, read_label_bounds, spatial_split

from .conftest import CHIP_NAME, square, write_chip, write_labels


def test_list_chip_paths_sorts_and_refuses_names_that_are_not_oam(tmp_path):
    folder = tmp_path / "chips"
    folder.mkdir()
    write_chip(folder / "OAM-2-0-3.tif")
    write_chip(folder / "OAM-1-0-3.tif")
    assert [path.name for path in list_chip_paths(folder)] == ["OAM-1-0-3.tif", "OAM-2-0-3.tif"]

    write_chip(folder / "tile_7.tif")
    with pytest.raises(ValueError, match="OAM"):
        list_chip_paths(folder)


def test_spatial_split_keeps_each_block_on_one_side():
    names = [f"OAM-{x}-{y}-5.tif" for x in range(8) for y in range(8)]
    train, val = spatial_split(names, 0.25, seed=1, block_size=4)
    assert sorted(train + val) == sorted(names)
    assert len(val) == 16  # whole 4 x 4 blocks of 16 chips each
    for block_x in (0, 1):
        for block_y in (0, 1):
            members = {
                name
                for name in names
                if int(name.split("-")[1]) // 4 == block_x and int(name.split("-")[2]) // 4 == block_y
            }
            assert members <= set(train) or members <= set(val)


def test_spatial_split_is_repeatable_for_a_seed():
    names = [f"OAM-{x}-{y}-5.tif" for x in range(6) for y in range(6)]
    assert spatial_split(names, 0.2, seed=7) == spatial_split(names, 0.2, seed=7)


def test_spatial_split_refuses_a_name_that_is_not_oam():
    with pytest.raises(ValueError):
        spatial_split(["chip.tif"], 0.2, seed=0)


def test_label_bounds_keep_polygons_and_skip_points_and_lines(tmp_path):
    labels = write_labels(
        tmp_path / "labels.geojson",
        [
            (square(1, 1, 3, 3), {"leisure": "swimming_pool"}),
            ({"type": "Point", "coordinates": [5, 5]}, {"leisure": "swimming_pool"}),
            ({"type": "LineString", "coordinates": [[0, 0], [9, 9]]}, {}),
        ],
    )
    bounds, crs = read_label_bounds(labels)
    assert bounds.tolist() == [[1.0, 1.0, 3.0, 3.0]]
    assert crs == "EPSG:4326"


def test_chip_labels_clip_a_pool_at_the_edge_and_leave_negatives_empty(tmp_path):
    chips = tmp_path / "chips"
    chips.mkdir()
    write_chip(chips / CHIP_NAME, left=0.0, top=10.0)  # covers lon 0..10, lat 0..10
    write_chip(chips / "OAM-1-0-20.tif", left=100.0, top=10.0)  # far away
    labels = write_labels(
        tmp_path / "labels.geojson",
        [(square(8, 8, 12, 12), {"leisure": "swimming_pool"})],  # crosses the east and north edges
    )
    result = build_chip_labels(chips, labels)
    assert result[CHIP_NAME] == [{"x1": 8.0, "y1": 0.0, "x2": 10.0, "y2": 2.0, "class_id": 0}]
    assert result["OAM-1-0-20.tif"] == []


def test_a_box_thinner_than_min_box_px_is_dropped(tmp_path):
    chips = tmp_path / "chips"
    chips.mkdir()
    write_chip(chips / CHIP_NAME, left=0.0, top=10.0)
    labels = write_labels(tmp_path / "labels.geojson", [(square(1, 1, 1.5, 6), {})])
    assert build_chip_labels(chips, labels, min_box_px=2.0)[CHIP_NAME] == []
    assert len(build_chip_labels(chips, labels, min_box_px=0.0)[CHIP_NAME]) == 1


def test_labels_in_another_crs_are_reprojected_to_the_chip(tmp_path):
    chips = tmp_path / "chips"
    chips.mkdir()
    # Chip in Web Mercator: 10 m pixels, origin at (-50, 50) m, so it covers x -50..50 and y -50..50 m.
    write_chip(chips / CHIP_NAME, left=-50.0, top=50.0, res=10.0, crs="EPSG:3857")
    labels = tmp_path / "labels.geojson"
    labels.write_text(
        '{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}, '
        '"features": [{"type": "Feature", "properties": {}, "geometry": '
        '{"type": "Polygon", "coordinates": [[[-20, -20], [20, -20], [20, 20], [-20, 20], [-20, -20]]]}}]}',
        encoding="utf-8",
    )
    (box,) = build_chip_labels(chips, labels, min_box_px=0.0)[CHIP_NAME]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == pytest.approx((3.0, 3.0, 7.0, 7.0))
