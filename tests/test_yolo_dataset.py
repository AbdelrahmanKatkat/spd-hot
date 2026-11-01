import pytest

from spd_hot.yolo_dataset import remove_yolo_dataset, write_yolo_dataset, yolo_label_line

from .conftest import CHIP_NAME


def test_a_box_becomes_a_normalised_yolo_line():
    box = {"x1": 2.0, "y1": 4.0, "x2": 6.0, "y2": 8.0, "class_id": 0}
    assert yolo_label_line(box, 10, 20) == "0 0.400000 0.300000 0.400000 0.200000"


def test_an_empty_box_gives_no_line():
    assert yolo_label_line({"x1": 3.0, "y1": 3.0, "x2": 3.0, "y2": 5.0, "class_id": 0}, 10, 10) is None


def test_the_folder_has_links_labels_and_data_yaml(chips_dir, tmp_path):
    out = write_yolo_dataset(
        chips_dir,
        {CHIP_NAME: [{"x1": 2.0, "y1": 2.0, "x2": 4.0, "y2": 4.0, "class_id": 0}]},
        train_names=[CHIP_NAME],
        val_names=[],
        out_dir=tmp_path / "yolo",
    )
    assert (out / "images" / "train" / CHIP_NAME).is_file()
    label = (out / "labels" / "train" / "OAM-524288-524288-20.txt").read_text(encoding="utf-8")
    assert label == "0 0.300000 0.300000 0.200000 0.200000"
    assert (out / "labels" / "val").is_dir()
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text and "names: ['swimming_pool']" in yaml_text


def test_a_chip_without_pools_gets_an_empty_label_file(chips_dir, tmp_path):
    out = write_yolo_dataset(chips_dir, {}, train_names=[CHIP_NAME], val_names=[], out_dir=tmp_path / "yolo")
    assert (out / "labels" / "train" / "OAM-524288-524288-20.txt").read_text(encoding="utf-8") == ""


def test_a_non_empty_folder_is_never_overwritten(chips_dir, tmp_path):
    existing = tmp_path / "yolo"
    existing.mkdir()
    (existing / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_yolo_dataset(chips_dir, {}, train_names=[CHIP_NAME], val_names=[], out_dir=existing)
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_removing_the_dataset_keeps_the_source_chips(chips_dir, tmp_path):
    out = write_yolo_dataset(chips_dir, {}, train_names=[CHIP_NAME], val_names=[], out_dir=tmp_path / "yolo")
    assert (out / "images" / "train" / CHIP_NAME).is_file()

    remove_yolo_dataset(out)

    assert not out.exists()
    assert (chips_dir / CHIP_NAME).is_file()


def test_a_folder_that_is_not_a_yolo_dataset_is_never_removed(tmp_path):
    other = tmp_path / "data"
    other.mkdir()
    (other / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        remove_yolo_dataset(other)
    assert (other / "keep.txt").read_text(encoding="utf-8") == "preserve"
