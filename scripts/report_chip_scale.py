"""Report each OAM chip's true ground size and the ground size of one model pixel (to check the resize against the scale).

python scripts/report_chip_scale.py data/pools/chips --model-input-size 640
"""

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from spd_hot.dataset import list_chip_paths
from spd_hot.params import MODEL_INPUT_SIZE
from spd_hot.tiles import tile_footprint_m, tile_gsd_m


def _chip_record(path: Path, model_input_size: int) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    footprint_w, footprint_h = tile_footprint_m(path.name, width, height)
    return {
        "chip": path.name,
        "width_px": width,
        "height_px": height,
        "gsd_m_px": tile_gsd_m(path.name, width),
        "ground_width_m": footprint_w,
        "ground_height_m": footprint_h,
        "model_gsd_m_px": footprint_w / model_input_size,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("chips", type=Path)
    parser.add_argument("--model-input-size", type=int, default=MODEL_INPUT_SIZE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    records = [_chip_record(path, args.model_input_size) for path in list_chip_paths(args.chips)]
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
