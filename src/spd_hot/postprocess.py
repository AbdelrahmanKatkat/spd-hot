"""Postprocessing: ONNX tensor → detections, then detections → GeoJSON."""

from typing import Any, Mapping

from .decode import decode_yolo_onnx_output
from .georef import area_xyxy, clip_xyxy_to_image, unscale_from_model_space, xyxy_to_epsg4326_polygon
from .params import PostprocessParams, parse_postprocess_params
from .preprocess import PreprocessMeta


def postprocess_onnx_output(
    raw_output: Any,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int]]:
    """Decode a YOLO ONNX tensor into detections (confidence filter + NMS).

    This is the STAC-facing postprocess step: it does not georeference.
    Map-space conversion is `detections_to_feature_collection`.
    """
    cfg = parse_postprocess_params(params)
    return decode_yolo_onnx_output(
        raw_output,
        confidence_threshold=cfg.confidence_threshold,
        iou_threshold=cfg.iou_threshold,
    )


def detections_to_feature_collection(
    detections: list[dict[str, float | int]],
    meta: PreprocessMeta,
    cfg: PostprocessParams,
    *,
    source_name: str | None = None,
    class_name: str | None = None,
) -> dict[str, Any]:
    """Convert decoded detection dicts into a GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = unscale_from_model_space(
            float(det["x1"]),
            float(det["y1"]),
            float(det["x2"]),
            float(det["y2"]),
            meta,
        )
        x1, y1, x2, y2 = clip_xyxy_to_image(x1, y1, x2, y2, meta.width, meta.height)
        if area_xyxy(x1, y1, x2, y2) < cfg.min_box_area_px:
            continue

        polygon = xyxy_to_epsg4326_polygon(x1, y1, x2, y2, meta.transform, meta.crs)
        props: dict[str, Any] = {
            "confidence": round(float(det["confidence"]), 4),
            "class_id": int(det["class_id"]),
        }
        if class_name:
            props["class_name"] = class_name
        if source_name:
            props["source"] = source_name

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        )

    return {"type": "FeatureCollection", "features": features}

