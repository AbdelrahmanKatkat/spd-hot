from rasterio.transform import from_origin

from spd_hot.postprocess import detections_to_feature_collection
from spd_hot.preprocess import PreprocessMeta


def _meta():
    return PreprocessMeta(
        width=10,
        height=10,
        resized_width=640,
        resized_height=640,
        transform=from_origin(0, 10, 1, 1),
        crs="EPSG:4326",
    )


def test_feature_collection_has_polygons_and_the_pool_properties():
    dets = [{"x1": 2.0, "y1": 2.0, "x2": 4.0, "y2": 4.0, "confidence": 0.123456, "class_id": 0}]
    collection = detections_to_feature_collection(dets, _meta(), source_name="OAM-1-2-3.tif")
    assert collection["type"] == "FeatureCollection"
    (feature,) = collection["features"]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"] == {
        "confidence": 0.1235,
        "class_id": 0,
        "class_name": "swimming_pool",
        "source": "OAM-1-2-3.tif",
    }


def test_empty_detections_give_an_empty_collection():
    assert detections_to_feature_collection([], _meta()) == {"type": "FeatureCollection", "features": []}


def test_boxes_outside_the_chip_are_dropped():
    dets = [{"x1": 20.0, "y1": 20.0, "x2": 30.0, "y2": 30.0, "confidence": 0.9, "class_id": 0}]
    assert detections_to_feature_collection(dets, _meta())["features"] == []
