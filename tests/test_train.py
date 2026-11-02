import math
import sys
from types import SimpleNamespace

import pytest

from spd_hot.params import PreprocessParams, TrainParams
from spd_hot.train import _epoch_losses, _epoch_report, _resolve_device, train


def test_train_hands_the_optimizer_and_learning_rate_to_ultralytics(monkeypatch, tmp_path):
    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def add_callback(self, *_args):
            pass

        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr("spd_hot.train._resolve_device", lambda requested: "cpu")
    train(
        weights="base.pt",
        dataset_yaml=tmp_path / "data.yaml",
        cfg=TrainParams(learning_rate=0.003),
        pre=PreprocessParams(),
        project_dir=tmp_path,
    )

    # An explicit optimizer keeps lr0: Ultralytics' `auto` would ignore it.
    assert calls["train"]["optimizer"] == "AdamW"
    assert calls["train"]["lr0"] == 0.003
    assert calls["train"]["imgsz"] == 512
    # Mozilla's training settings, with the first 10 layers frozen.
    sent = calls["train"]
    assert (sent["batch"], sent["epochs"], sent["patience"], sent["cos_lr"]) == (16, 10, 3, True)
    assert (sent["flipud"], sent["scale"], sent["translate"], sent["freeze"]) == (0.5, 0.0, 0.0, 10)


def _fake_torch(*, cuda: bool):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("auto", "0"), ("cuda", "0"), ("gpu", "0"), ("CUDA", "0"), ("1", "1"), ("0,1", "0,1"), ("cpu", "cpu")],
)
def test_a_visible_gpu_is_used_unless_the_cpu_is_asked_for(monkeypatch, requested, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert _resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["auto", "cuda", "gpu", "0", "cpu"])
def test_without_a_gpu_training_falls_back_to_the_cpu(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert _resolve_device(requested) == "cpu"


def test_train_sends_the_gpu_to_ultralytics_when_one_is_visible(monkeypatch, tmp_path):
    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def add_callback(self, *_args):
            pass

        def train(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    train(
        weights="base.pt",
        dataset_yaml=tmp_path / "data.yaml",
        cfg=TrainParams(),
        pre=PreprocessParams(),
        project_dir=tmp_path,
    )

    assert calls["device"] == "0"


def test_the_epoch_losses_come_from_the_trainer_dicts():
    trainer = SimpleNamespace(
        tloss={"box": 0.5, "cls": 0.25, "dfl": 0.25},
        metrics={"val/box_loss": 0.6, "val/cls_loss": 0.2, "val/dfl_loss": 0.2, "metrics/mAP50(B)": 0.7},
    )
    train_loss, val_loss = _epoch_losses(trainer)
    assert train_loss == 1.0
    assert val_loss == 1.0


def test_no_validation_metrics_give_no_validation_loss():
    trainer = SimpleNamespace(tloss={"box": 0.5}, metrics={"metrics/mAP50(B)": 0.7})
    assert _epoch_losses(trainer) == (0.5, None)


def test_a_missing_train_loss_is_nan_not_an_error():
    train_loss, _ = _epoch_losses(SimpleNamespace(tloss=None, metrics={}))
    assert math.isnan(train_loss)


def test_each_epoch_reports_to_the_callback_with_its_number():
    seen = []
    report = _epoch_report(3, lambda epoch, train, val: seen.append((epoch, train, val)))
    report(
        SimpleNamespace(
            epoch=0,
            tloss={"box": 0.5, "cls": 0.5},
            metrics={"val/box_loss": 0.4},
        )
    )
    assert seen == [(1, 1.0, 0.4)]
