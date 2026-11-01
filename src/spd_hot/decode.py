"""ONNX output -> detections in model-input pixels (no CRS or affine dependence).

The graph returns `output0` with shape `(1, 4 + nc, anchors)`: for every anchor `cx, cy, w, h` in model-input pixels,
then one sigmoid score per class. Decoding here is the confidence filter, per-class NMS and the per-chip cap.
"""

from typing import Any

import numpy as np

from .params import MAX_DETECTIONS, PostprocessParams


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on XYXY boxes; returns kept indices in descending score order."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = overlap_w * overlap_h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_detections(output: Any, cfg: PostprocessParams | None = None) -> list[dict[str, float | int]]:
    """`output0` of one chip -> detections `x1, y1, x2, y2, confidence, class_id`, best first, in model-input pixels.

    Raises `ValueError` for any other layout, instead of guessing which axis holds the features.
    """
    cfg = cfg or PostprocessParams()
    preds = np.asarray(output, dtype=np.float32)
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]
    # The features are the short axis: a transposed (1, anchors, 4 + nc) tensor is refused, not read as features.
    if preds.ndim != 2 or preds.shape[0] < 5 or preds.shape[0] > preds.shape[1]:
        raise ValueError(f"expected YOLO output (1, 4 + nc, anchors), got shape {np.asarray(output).shape}")

    cx, cy, w, h = preds[:4]
    class_scores = preds[4:].T
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep = scores > cfg.confidence_threshold
    cx, cy, w, h = cx[keep], cy[keep], w[keep], h[keep]
    scores, class_ids = scores[keep], class_ids[keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        members = np.flatnonzero(class_ids == class_id)
        kept.extend(int(members[i]) for i in nms(boxes[members], scores[members], cfg.nms_threshold))
    kept.sort(key=lambda i: -scores[i])
    return [
        {
            "x1": float(boxes[i, 0]),
            "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]),
            "y2": float(boxes[i, 3]),
            "confidence": float(scores[i]),
            "class_id": int(class_ids[i]),
        }
        for i in kept[:MAX_DETECTIONS]
    ]
