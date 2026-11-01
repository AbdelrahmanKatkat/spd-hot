import pytest

from spd_hot.evaluate import match_detections, score_session, sweep_confidence
from spd_hot.params import MODEL_INPUT_SIZE

from .conftest import CHIP_NAME, FakeSession, yolo_output

TRUTH = {"x1": 4.0, "y1": 4.0, "x2": 6.0, "y2": 6.0, "class_id": 0}
SCALE = MODEL_INPUT_SIZE / 10  # the test chip is 10 px, so one chip pixel is SCALE model pixels
HALF = MODEL_INPUT_SIZE / 2  # the chip centre (5, 5) in model pixels


def _det(box, confidence):
    return {**box, "confidence": confidence, "class_id": 0}


def test_match_is_greedy_by_confidence_and_one_to_one():
    preds = [_det(TRUTH, 0.9), _det(TRUTH, 0.8)]
    assert match_detections(preds, [TRUTH]) == (1, 1, 0)
    assert match_detections([], [TRUTH]) == (0, 0, 1)
    assert match_detections([_det(TRUTH, 0.9)], []) == (0, 1, 0)


def test_score_session_compares_chip_pixels_not_model_pixels(chips_dir):
    # The chip is 10 px and the model MODEL_INPUT_SIZE px: a box centred at (5, 5) in chip pixels is at half the model
    # size in model pixels, and the 2 px truth box is 2 * SCALE wide. Scoring must map it back to chip pixels before it
    # is matched with the truth box.
    session = FakeSession(yolo_output([(HALF, HALF, 2 * SCALE, 2 * SCALE)], [0.9]))
    truth = {CHIP_NAME: [TRUTH]}
    scores = score_session(session, images_dir=chips_dir, labels=truth, iou_thresholds=(0.5,))
    assert scores[0.5]["tp"] == 1.0
    assert scores[0.5]["precision"] == pytest.approx(1.0)
    assert scores[0.5]["recall"] == pytest.approx(1.0)


def test_score_session_only_scores_the_named_chips(chips_dir):
    session = FakeSession(yolo_output([], []))
    scores = score_session(session, images_dir=chips_dir, labels={CHIP_NAME: [TRUTH]}, names=["other.tif"])
    assert scores[0.5]["fn"] == 0.0 and scores[0.5]["tp"] == 0.0


def test_the_sweep_shows_the_precision_gain_of_a_higher_threshold(chips_dir):
    output = yolo_output(
        [(HALF, HALF, 2 * SCALE, 2 * SCALE), (1.5 * SCALE, 1.5 * SCALE, SCALE, SCALE)],
        [0.9, 0.3],
    )
    # The second box lands at chip (1.5, 1.5), away from the only pool: a false positive at the lower threshold.
    rows = sweep_confidence(
        FakeSession(output),
        images_dir=chips_dir,
        labels={CHIP_NAME: [TRUTH]},
        thresholds=(0.2, 0.5),
    )
    low, high = rows
    assert low["fp"] == 1.0 and high["fp"] == 0.0
    assert high["precision"] == pytest.approx(1.0)
    assert high["detections"] == 1.0
