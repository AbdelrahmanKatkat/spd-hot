import pytest
from rasterio.transform import from_origin

from spd_hot.georef import clip_xyxy_to_image, detections_to_chip_pixels, xyxy_to_epsg4326_polygon
from spd_hot.preprocess import PreprocessMeta


def _meta(width=10, height=10, resized=640, crs="EPSG:4326"):
    return PreprocessMeta(
        width=width,
        height=height,
        resized_width=resized,
        resized_height=resized,
        transform=from_origin(0, 10, 1, 1),
        crs=crs,
    )


def test_model_pixels_are_mapped_back_to_chip_pixels():
    (det,) = detections_to_chip_pixels(
        [{"x1": 320.0, "y1": 0.0, "x2": 640.0, "y2": 64.0, "confidence": 0.5, "class_id": 0}], _meta()
    )
    assert (det["x1"], det["y1"], det["x2"], det["y2"]) == pytest.approx((5.0, 0.0, 10.0, 1.0))
    assert det["confidence"] == 0.5


def test_boxes_are_clipped_to_the_chip():
    assert clip_xyxy_to_image(-2, -1, 12, 5, 10, 10) == (0.0, 0.0, 10.0, 5.0)


def test_epsg4326_chip_gives_a_closed_lon_lat_ring():
    ring = xyxy_to_epsg4326_polygon(1, 2, 3, 4, from_origin(0, 10, 1, 1), "EPSG:4326")
    assert ring[0] == [1.0, 8.0] and ring[2] == [3.0, 6.0] and ring[0] == ring[-1]


def test_projected_chip_is_reprojected_to_lon_lat():
    # Web Mercator metres at the origin are lon 0, lat 0.
    ring = xyxy_to_epsg4326_polygon(0, 0, 1, 1, from_origin(0, 0, 1, 1), "EPSG:3857")
    assert ring[0] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert ring[2][0] == pytest.approx(1 / 111319.49, rel=1e-3)
