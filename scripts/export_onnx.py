"""Entry point to `spd_hot.export`: write a fine-tuned YOLO checkpoint as an ONNX file.

    python scripts/export_onnx.py data/runs/pools_v1/swimming_pool.pt data/runs/pools_v1/swimming_pool.onnx

Needs the `train` extra (torch, ultralytics and onnx); the output runs with `scripts/predict.py` or `spd_hot.predict_session`.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from spd_hot.export import export_onnx_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", type=Path, help="a YOLO checkpoint (.pt)")
    parser.add_argument("out", type=Path, help="the ONNX file to write")
    return parser


def _export(weights: Path, out: Path) -> None:
    out.write_bytes(export_onnx_bytes(YOLO(str(weights))))


def main() -> None:
    args = _build_parser().parse_args()
    _export(args.weights, args.out)


if __name__ == "__main__":
    main()
