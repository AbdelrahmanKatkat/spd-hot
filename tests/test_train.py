import math
from types import SimpleNamespace

from spd_hot.train import _epoch_losses, _epoch_report


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
