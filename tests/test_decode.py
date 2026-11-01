import numpy as np
import pytest

from spd_hot.decode import decode_detections, nms
from spd_hot.params import PostprocessParams

from .conftest import yolo_output


def test_nms_keeps_the_best_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert nms(boxes, scores, iou_threshold=0.5) == [1, 2]


def test_one_anchor_above_the_threshold_gives_one_box_in_xyxy():
    output = yolo_output([(320.0, 320.0, 100.0, 120.0)], [0.9])
    (det,) = decode_detections(output)
    assert (det["x1"], det["y1"], det["x2"], det["y2"]) == pytest.approx((270.0, 260.0, 370.0, 380.0))
    assert det["confidence"] == pytest.approx(0.9)
    assert det["class_id"] == 0


def test_scores_at_or_below_the_threshold_are_dropped():
    output = yolo_output([(320.0, 320.0, 100.0, 120.0)], [0.1])
    assert decode_detections(output, PostprocessParams(confidence_threshold=0.25)) == []


def test_overlapping_anchors_are_suppressed_by_nms():
    output = yolo_output([(320.0, 320.0, 100.0, 100.0), (322.0, 320.0, 100.0, 100.0)], [0.9, 0.8])
    assert len(decode_detections(output)) == 1


def test_a_transposed_output_is_refused_instead_of_read():
    transposed = np.zeros((1, 4000, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="1, 4 \\+ nc, anchors"):
        decode_detections(transposed)
