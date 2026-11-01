"""Entry point to `spd_hot.train`: fine-tune a YOLO11 checkpoint on chips + GeoJSON pools; write `<name>.pt` and `<name>.onnx`.

    python scripts/finetune.py --chips data/pools/chips --labels data/pools/labels.geojson \
        --weights weights/model.pt --out data/runs/pools_v1 --name swimming_pool --epochs 20

The chips are georeferenced GeoTIFFs named `OAM-<x>-<y>-<z>.tif`; the labels are one GeoJSON of pool polygons (see
`research/analysis/make_pool_labels.py`). The run folder gets `split.json` (the train and validation chips),
`train.log`, `<name>.pt`, `<name>.onnx` and Ultralytics' `runs/`. The Ultralytics dataset (`yolo/`) holds only links to
the chips and small label files; it is removed when the run succeeds, unless `--keep-dataset` is given. Options left
out keep the defaults of `spd_hot.params`. Needs the `train` extra; with a local CUDA torch use the venv python, not a
plain `uv run`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from spd_hot.dataset import build_chip_labels, spatial_split
from spd_hot.export import export_onnx_bytes
from spd_hot.params import (
    PreprocessParams,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from spd_hot.train import train
from spd_hot.yolo_dataset import remove_yolo_dataset, write_yolo_dataset

log = logging.getLogger("finetune")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument(
        "--labels", required=True, type=Path, help="one GeoJSON of pool polygons, or a folder holding one"
    )
    parser.add_argument("--weights", required=True, type=Path, help="the checkpoint to start from (.pt)")
    parser.add_argument("--out", required=True, type=Path, help="output folder; must be empty or new")
    parser.add_argument("--name", default="swimming_pool", help="output file stem: <name>.pt and <name>.onnx")
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, help="learning rate of Ultralytics' optimizer (lr0)")
    parser.add_argument("--device", type=str, help="auto, cpu, cuda or a GPU index")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--keep-dataset", action="store_true", help="keep <out>/yolo (links and labels) after the run"
    )
    return parser


def _prepare_out_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)


def _configure_logging(out: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "train.log", encoding="utf-8"),
        ],
    )


def _split_hyperparameters(args: argparse.Namespace) -> dict[str, object]:
    options = {"val_ratio": args.val_ratio, "split_seed": args.split_seed, "block_size": args.block_size}
    return {key: value for key, value in options.items() if value is not None}


def _train_hyperparameters(args: argparse.Namespace) -> dict[str, object]:
    options = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "freeze_encoder": args.freeze_encoder,
    }
    return {key: value for key, value in options.items() if value is not None}


def _split_chips(args: argparse.Namespace) -> tuple[dict, list[str], list[str]]:
    """Pixel boxes of every chip, and the spatial train / validation split of the chip names."""
    split = parse_split_params(_split_hyperparameters(args))
    labels = build_chip_labels(args.chips, args.labels)
    train_names, val_names = spatial_split(
        sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size
    )
    return labels, train_names, val_names


def _train_model(
    args: argparse.Namespace,
    out: Path,
    labels: dict[str, list[dict]],
    train_names: list[str],
    val_names: list[str],
    pre: PreprocessParams,
):
    yolo_dir = write_yolo_dataset(
        args.chips,
        labels,
        train_names=train_names,
        val_names=val_names,
        out_dir=out / "yolo",
    )
    return train(
        weights=args.weights,
        dataset_yaml=yolo_dir / "data.yaml",
        cfg=parse_train_params(_train_hyperparameters(args)),
        pre=pre,
        project_dir=out / "runs",
        run_name=args.name,
    )


def _save_outputs(model, out: Path, name: str, pre: PreprocessParams) -> None:
    """The checkpoint (`.pt`) and the ONNX export, and one log line naming the files."""
    pt_path, onnx_path = out / f"{name}.pt", out / f"{name}.onnx"
    model.save(pt_path)
    onnx_path.write_bytes(export_onnx_bytes(model, imgsz=pre.model_input_size))
    log.info(
        "wrote %s (%.1f MB) and %s (%.1f MB)",
        pt_path,
        pt_path.stat().st_size / 1e6,
        onnx_path,
        onnx_path.stat().st_size / 1e6,
    )


def main() -> None:
    args = _build_parser().parse_args()
    out = args.out
    _prepare_out_dir(out)
    _configure_logging(out)
    pre = parse_preprocess_params(None)
    labels, train_names, val_names = _split_chips(args)
    (out / "split.json").write_text(
        json.dumps({"train": train_names, "val": val_names}, indent=2), encoding="utf-8"
    )
    pools = sum(1 for boxes in labels.values() if boxes)
    log.info(
        "chips: %d with pools, %d without; split: %d train, %d val",
        pools,
        len(labels) - pools,
        len(train_names),
        len(val_names),
    )
    model = _train_model(args, out, labels, train_names, val_names, pre)
    _save_outputs(model, out, args.name, pre)
    if not args.keep_dataset:  # only after a successful run: a failed run keeps the folder to debug with
        remove_yolo_dataset(out / "yolo")
        log.info("removed %s (links and labels only; the chips are untouched)", out / "yolo")


if __name__ == "__main__":
    main()
