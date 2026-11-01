"""`spd-hot`: HOT fAIr swimming-pool detector helpers (the inference path; training is imported explicitly)."""

from .dataset import build_chip_labels, spatial_split
from .decode import decode_detections, nms
from .params import (
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from .postprocess import detections_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip, preprocess_rgb
from .serve import predict_chip, predict_session
from .tiles import tile_gsd_m

__all__ = [
    "PostprocessParams",
    "PreprocessMeta",
    "PreprocessParams",
    "SplitParams",
    "TrainParams",
    "build_chip_labels",
    "decode_detections",
    "detections_to_feature_collection",
    "nms",
    "parse_postprocess_params",
    "parse_preprocess_params",
    "parse_split_params",
    "parse_train_params",
    "predict_chip",
    "predict_session",
    "preprocess_chip",
    "preprocess_rgb",
    "spatial_split",
    "tile_gsd_m",
]
# Optional (train extra, import explicitly): spd_hot.yolo_dataset, spd_hot.train, spd_hot.export
