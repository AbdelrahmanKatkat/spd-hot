"""Precision, recall and F1 of a swimming-pool detector, through the same ONNX path that serves it (no torch).

Predictions are mapped back to chip pixels (`serve.detect_chip`) before they are matched against the labels, which are
chip-pixel boxes from `dataset.build_chip_labels`.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .serve import detect_chip, list_images

log = logging.getLogger(__name__)


def _iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return float(inter / (area_a + area_b - inter + 1e-9))


def match_detections(
    preds: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy matching by confidence, one label per detection: returns `(tp, fp, fn)`."""
    if not preds and not gts:
        return 0, 0, 0
    pred_boxes = [
        np.asarray([pred["x1"], pred["y1"], pred["x2"], pred["y2"]], dtype=np.float32)
        for pred in sorted(preds, key=lambda det: -float(det.get("confidence", 0.0)))
    ]
    gt_boxes = [np.asarray([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], dtype=np.float32) for gt in gts]
    used = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if used[gt_index]:
                continue
            iou = _iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_iou >= float(iou_threshold) and best_index >= 0:
            used[best_index] = True
            tp += 1
        else:
            fp += 1
    fn = int(len(gt_boxes) - sum(used))
    return int(tp), int(fp), int(fn)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _chips(images_dir: str | Path, names: Sequence[str] | None) -> list[Path]:
    """The chips of a folder, or only those named (for example a validation split)."""
    paths = list_images(images_dir)
    if names is None:
        return paths
    wanted = set(names)
    return [path for path in paths if path.name in wanted]


def _score(
    paths: Sequence[Path],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    predict: Callable[[Path], list[dict[str, float | int]]],
    iou_thresholds: Sequence[float],
) -> dict[float, dict[str, float]]:
    results: dict[float, dict[str, float]] = {}
    for iou in iou_thresholds:
        tally = [0, 0, 0]  # tp, fp, fn
        for path in paths:
            tp, fp, fn = match_detections(predict(path), labels.get(path.name, []), iou_threshold=iou)
            tally[0] += tp
            tally[1] += fp
            tally[2] += fn
        results[iou] = _prf(*tally)
    return results


def score_session(
    session: Any,
    *,
    images_dir: str | Path,
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
    iou_thresholds: tuple[float, ...] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Precision, recall and F1 at each IoU threshold, for the chips in `images_dir` (or only those in `names`)."""
    paths = _chips(images_dir, names)
    return _score(paths, labels, lambda path: detect_chip(session, path, params)[0], iou_thresholds)


def sweep_confidence(
    session: Any,
    *,
    images_dir: str | Path,
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str] | None = None,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.25, 0.4, 0.5),
    iou_threshold: float = 0.5,
    params: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Precision, recall and F1 at each confidence threshold, from one prediction per chip.

    Predicts at the lowest threshold and filters upward. That is exact: NMS never lets a lower-scoring box suppress a
    higher-scoring one, and the per-chip cap keeps the best scores either way. Use it to choose the confidence threshold
    of a freshly trained model, whose scores are not on the scale of the base model's.
    """
    low_params = {**(params or {}), "confidence_threshold": min(thresholds)}
    predictions = {path.name: detect_chip(session, path, low_params)[0] for path in _chips(images_dir, names)}
    rows = []
    for threshold in thresholds:
        tally = [0, 0, 0, 0]  # tp, fp, fn, detections
        for name, preds in predictions.items():
            kept = [det for det in preds if det["confidence"] > threshold]
            for index, value in enumerate(
                match_detections(kept, labels.get(name, []), iou_threshold=iou_threshold)
            ):
                tally[index] += value
            tally[3] += len(kept)
        rows.append({"threshold": threshold, "detections": float(tally[3]), **_prf(*tally[:3])})
    best = max(rows, key=lambda row: row["f1"])
    log.info(
        "confidence sweep at IoU>=%.2f: best f1=%.3f at confidence %.2f",
        iou_threshold,
        best["f1"],
        best["threshold"],
    )
    return rows
