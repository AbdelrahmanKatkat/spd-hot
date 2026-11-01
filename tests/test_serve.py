import pytest

from spd_hot.params import MODEL_INPUT_SIZE
from spd_hot.serve import list_images, predict_session

from .conftest import CHIP_NAME, FakeSession, yolo_output


def test_predict_session_maps_a_model_space_box_back_to_the_map(chips_dir):
    # A 10 px chip is scaled to MODEL_INPUT_SIZE px. A box centred at half that size, one chip pixel wide, sits at
    # chip (5, 5).
    session = FakeSession(
        yolo_output(
            [(MODEL_INPUT_SIZE / 2, MODEL_INPUT_SIZE / 2, MODEL_INPUT_SIZE / 10, MODEL_INPUT_SIZE / 10)],
            [0.9],
        )
    )
    collection = predict_session(session, chips_dir, None)
    (feature,) = collection["features"]
    lons = [point[0] for point in feature["geometry"]["coordinates"][0]]
    lats = [point[1] for point in feature["geometry"]["coordinates"][0]]
    assert min(lons) == pytest.approx(4.5) and max(lons) == pytest.approx(5.5)
    assert min(lats) == pytest.approx(4.5) and max(lats) == pytest.approx(5.5)
    assert feature["properties"]["source"] == CHIP_NAME


def test_the_model_gets_one_batch_of_three_channels_at_the_input_size(chips_dir):
    session = FakeSession(yolo_output([], []))
    predict_session(session, chips_dir, {"model_input_size": 320})
    (batch,) = session.inputs
    assert batch.shape == (1, 3, 320, 320)
    assert batch.dtype.name == "float32"


def test_a_low_confidence_box_is_not_returned(chips_dir):
    session = FakeSession(yolo_output([(320.0, 320.0, 64.0, 64.0)], [0.1]))
    assert predict_session(session, chips_dir, None)["features"] == []


def test_a_threshold_from_params_is_applied(chips_dir):
    session = FakeSession(yolo_output([(320.0, 320.0, 64.0, 64.0)], [0.4]))
    assert predict_session(session, chips_dir, {"confidence_threshold": 0.5})["features"] == []


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)
