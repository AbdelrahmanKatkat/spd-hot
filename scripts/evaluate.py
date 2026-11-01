"""Entry point to `spd_hot.evaluate`: precision, recall and F1 of an ONNX detector on chips and GeoJSON pools (CPU, no torch).

    python scripts/evaluate.py --onnx data/runs/pools_v1/swimming_pool.onnx --chips data/pools/chips \
        --labels data/pools/labels.geojson --split data/runs/pools_v1/split.json --sweep

`--split` scores only the validation chips of a finetune run. `--sweep` prints the scores at several confidence
thresholds. Needs `onnxruntime` (the `infer` extra).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort

from spd_hot.dataset import build_chip_labels
from spd_hot.evaluate import score_session, sweep_confidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="one GeoJSON of pool polygons")
    parser.add_argument("--split", type=Path, help="split.json of a finetune run: score only its val chips")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true", help="scores at several confidence thresholds")
    parser.add_argument("--params", default="{}", help='JSON, for example {"nms_threshold": 0.3}')
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _validation_names(split: Path | None) -> list[str] | None:
    if split is None:
        return None
    return json.loads(split.read_text(encoding="utf-8"))["val"]


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    labels = build_chip_labels(args.chips, args.labels)
    names = _validation_names(args.split)
    params = json.loads(args.params)
    if args.sweep:
        rows = sweep_confidence(
            session, images_dir=args.chips, labels=labels, names=names, iou_threshold=args.iou, params=params
        )
        return {"sweep": rows}
    scores = score_session(
        session, images_dir=args.chips, labels=labels, names=names, params=params, iou_thresholds=(args.iou,)
    )
    return {"scores": {str(iou): row for iou, row in scores.items()}}


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging()
    print(json.dumps(_evaluate(args), indent=2))


if __name__ == "__main__":
    main()
