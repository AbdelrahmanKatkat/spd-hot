"""Model-specific prediction logic for fAIr ONNX serving."""

from pathlib import Path
from typing import Any

from .decode import decode_yolo_onnx_output
from .params import parse_inference_params, parse_postprocess_params, parse_preprocess_params
from .postprocess import detections_to_feature_collection
from .preprocess import preprocess_chip_for_onnx


def _iter_image_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(p for pattern in patterns for p in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def predict_session(session: Any, input_images: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Run ONNX prediction for all images in a directory.

    This function is intended to be called from `models/<name>/pipeline.py::predict`.
    """
    preprocess_cfg = parse_preprocess_params(params)
    infer_cfg = parse_inference_params(params)
    post_cfg = parse_postprocess_params(params)
    input_name = session.get_inputs()[0].name

    all_features: list[dict[str, Any]] = []
    for img_path in _iter_image_paths(input_images):
        batch, meta = preprocess_chip_for_onnx(img_path, preprocess_cfg)
        output = session.run(None, {input_name: batch})[0]
        detections = decode_yolo_onnx_output(
            output,
            confidence_threshold=infer_cfg.confidence_threshold,
            iou_threshold=infer_cfg.iou_threshold,
        )
        fc = detections_to_feature_collection(
            detections,
            meta,
            post_cfg,
            source_name=img_path.name,
            class_name="swimming_pool",
        )
        all_features.extend(fc["features"])

    return {"type": "FeatureCollection", "features": all_features}

