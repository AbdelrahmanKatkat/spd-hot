import pytest

from spd_hot.params import (
    MODEL_INPUT_SIZE,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)


def test_none_and_missing_keys_give_the_defaults():
    assert parse_split_params(None).block_size == 4
    assert parse_preprocess_params(None).model_input_size == MODEL_INPUT_SIZE
    post = parse_postprocess_params({})
    assert (post.confidence_threshold, post.nms_threshold) == (0.25, 0.45)
    train = parse_train_params(None)
    assert (train.epochs, train.freeze_encoder, train.device) == (10, True, "auto")


def test_the_optimizer_is_explicit_and_auto_is_refused():
    assert parse_train_params(None).optimizer == "AdamW"
    assert parse_train_params({"optimizer": "SGD"}).optimizer == "SGD"
    # Ultralytics' `auto` ignores the learning rate, so it is refused instead of silently dropping it.
    with pytest.raises(ValueError, match="auto"):
        parse_train_params({"optimizer": "auto"})


def test_aliases_are_read_and_canonical_names_win():
    assert parse_postprocess_params({"score_thresh": 0.4}).confidence_threshold == 0.4
    assert parse_postprocess_params({"iou_threshold": 0.3}).nms_threshold == 0.3
    assert parse_postprocess_params({"nms_thresh": 0.2}).nms_threshold == 0.2
    both = parse_postprocess_params({"confidence_threshold": 0.6, "score_thresh": 0.1})
    assert both.confidence_threshold == 0.6
    assert parse_preprocess_params({"chip_size": 512}).model_input_size == 512
    assert parse_train_params({"lr": 0.02}).learning_rate == 0.02
    assert parse_train_params({"freeze_backbone": False}).freeze_encoder is False


@pytest.mark.parametrize(
    ("parser", "params"),
    [
        (parse_split_params, {"val_ratio": 1}),
        (parse_split_params, {"block_size": 0}),
        (parse_preprocess_params, {"model_input_size": 630}),
        (parse_preprocess_params, {"model_input_size": 16}),
        (parse_postprocess_params, {"confidence_threshold": 2}),
        (parse_train_params, {"epochs": 0}),
        (parse_train_params, {"learning_rate": 0}),
    ],
)
def test_out_of_range_values_are_refused(parser, params):
    with pytest.raises(ValueError):
        parser(params)


def test_a_bool_is_not_a_number():
    with pytest.raises(TypeError):
        parse_preprocess_params({"model_input_size": True})
