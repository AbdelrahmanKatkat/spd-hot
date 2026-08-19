"""YOLO ONNX decoding helpers."""

from typing import Any

import numpy as np


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Perform standard NMS on XYXY boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolo_onnx_output(
    output: Any,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, float | int]]:
    """Decode YOLO ONNX output shaped like `(1, 4+nc, anchors)` or `(1, anchors, 4+nc)`."""
    preds = np.squeeze(np.asarray(output), axis=0)
    if preds.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape after squeeze: {preds.shape}")

    # Convert to (anchors, 4+nc)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    if class_scores.shape[1] == 0:
        return []

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    keep = confidences >= confidence_threshold
    if not keep.any():
        return []

    boxes_cxcywh = boxes_cxcywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep_idx = nms_xyxy(boxes_xyxy, confidences, iou_threshold)

    decoded: list[dict[str, float | int]] = []
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        decoded.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confidences[idx]),
                "class_id": int(class_ids[idx]),
            }
        )
    return decoded

