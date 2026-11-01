"""Precision, recall, F1 and mAP of a swimming-pool detector, through the same ONNX path that serves it (no torch).

Predictions are mapped back to chip pixels (`serve.detect_chip`) before they are matched against the labels, which are
chip-pixel boxes from `dataset.build_chip_labels`.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .params import MAP_CONFIDENCE, MAP_IOU_THRESHOLDS, MAP_NMS_IOU
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


def _hit_flags(
    preds: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]], iou_threshold: float
) -> list[tuple[float, bool]]:
    """`(confidence, is_true_positive)` of each detection of one chip: greedy by confidence, one label per detection."""
    gt_boxes = [np.asarray([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], dtype=np.float32) for gt in gts]
    used = [False] * len(gt_boxes)
    flags: list[tuple[float, bool]] = []
    for pred in sorted(preds, key=lambda det: -float(det.get("confidence", 0.0))):
        pred_box = np.asarray([pred["x1"], pred["y1"], pred["x2"], pred["y2"]], dtype=np.float32)
        best_iou, best_index = 0.0, -1
        for index, gt_box in enumerate(gt_boxes):
            if used[index]:
                continue
            iou = _iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_index = iou, index
        hit = best_index >= 0 and best_iou >= float(iou_threshold)
        if hit:
            used[best_index] = True
        flags.append((float(pred.get("confidence", 0.0)), hit))
    return flags


def match_detections(
    preds: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy matching by confidence, one label per detection: returns `(tp, fp, fn)`."""
    tp = sum(hit for _, hit in _hit_flags(preds, gts, iou_threshold))
    return tp, len(preds) - tp, len(gts) - tp


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


def _average_precision(flags: Sequence[tuple[float, bool]], n_labels: int) -> float:
    """Area under the precision-recall curve of detections ranked by confidence (101-point interpolation)."""
    if n_labels == 0 or not flags:
        return 0.0
    ranked = sorted(flags, key=lambda flag: -flag[0])
    hits = np.asarray([hit for _, hit in ranked], dtype=np.float64)
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(1.0 - hits)
    recall = true_positives / n_labels
    precision = true_positives / (true_positives + false_positives)
    # The curve ends at the highest recall reached: a second sentinel at that recall drops precision to 0 there, so
    # no area is added between it and recall 1 (as in Ultralytics' `compute_ap`).
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision never rises as recall grows
    grid = np.linspace(0.0, 1.0, 101)
    curve = np.interp(grid, mrec, mpre)
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(grid)))


def score_map(
    session: Any,
    *,
    images_dir: str | Path,
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
    confidence_threshold: float = MAP_CONFIDENCE,
    nms_threshold: float = MAP_NMS_IOU,
) -> dict[str, float]:
    """`{"map50": ..., "map50_95": ...}` of the chips in `images_dir` (or only those in `names`), with no torch.

    mAP50 is the average precision at IoU 0.5; mAP50-95 is its mean over IoU 0.50, 0.55, ..., 0.95. Every chip is
    predicted once, with the confidence and NMS settings of Ultralytics' validation. The matching (by confidence, one
    label per detection) and the average precision (101-point interpolation) are those of Ultralytics' validator: fed the
    same detections, `ultralytics.utils.metrics.ap_per_class` returns the same mAP, and a perfect detector scores 0.995
    in both. The detections themselves come from this library's resize (Pillow), not from Ultralytics' dataloader, so
    the two can differ on the same chips. The confidence and NMS settings in `params` are not used:
    a confidence filter at the serving value would cut the curve short. Labels are chip-pixel boxes, as in
    `score_session`.
    """
    paths = _chips(images_dir, names)
    predict_params = {
        **(params or {}),
        "confidence_threshold": confidence_threshold,
        "nms_threshold": nms_threshold,
    }
    predictions = {path.name: detect_chip(session, path, predict_params)[0] for path in paths}
    n_labels = sum(len(labels.get(name, [])) for name in predictions)
    if n_labels == 0:
        log.warning("no labelled pool in the %d scored chips: mAP is 0", len(paths))
    average_precisions = []
    for iou in MAP_IOU_THRESHOLDS:
        flags = [
            flag
            for name, preds in predictions.items()
            for flag in _hit_flags(preds, labels.get(name, []), iou)
        ]
        average_precisions.append(_average_precision(flags, n_labels))
    return {"map50": average_precisions[0], "map50_95": float(np.mean(average_precisions))}


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
