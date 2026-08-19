"""Spatial train/val split for OAM chips."""

import re

import numpy as np

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def spatial_split(
    chip_names: list[str],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; entire `(x//K, y//K)` blocks pick a side.

    Chips that do not match `OAM-x-y-z` are shuffled and used only to fill leftover val slots.
    """
    matched: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        oam_match = OAM_TILE_RE.match(name)
        if oam_match is None:
            unmatched.append(name)
            continue
        tile_x, tile_y = int(oam_match.group(1)), int(oam_match.group(2))
        matched.setdefault((tile_x // block_size, tile_y // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(matched.keys())
    rng.shuffle(blocks)

    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio)) if n_total else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        bucket = val if len(val) < n_val_target else train
        bucket.extend(matched[block])

    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        remaining = max(0, n_val_target - len(val))
        val.extend(leftover[:remaining])
        train.extend(leftover[remaining:])

    return sorted(train), sorted(val)
