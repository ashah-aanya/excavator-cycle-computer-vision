"""Storing masks compactly.

A boolean mask per sample is small in principle and large in practice: 870
samples of 480x272 is 113 MB raw, and that is the file every later stage has to
load. Run-length encoding takes it to a few MB, because a mask is mostly long
runs of False with a machine-shaped island of True in the middle.

Why not ``pycocotools``: it is a C extension, and this is twenty lines of NumPy
that needs no build step on a cluster node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def encode(mask: np.ndarray) -> np.ndarray:
    """Mask -> run lengths.

    The encoding is the lengths of alternating runs, always starting with a run
    of False (length 0 if the mask starts True). Shape is stored separately.
    """
    flat = mask.reshape(-1).astype(np.uint8)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)

    # Positions where the value changes, plus the ends, give the run boundaries.
    change = np.flatnonzero(np.diff(flat)) + 1
    edges = np.concatenate(([0], change, [flat.size]))
    lengths = np.diff(edges)

    if flat[0]:  # ensure the first run is always the False run
        lengths = np.concatenate(([0], lengths))
    return lengths.astype(np.int64)


def decode(lengths: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Run lengths -> mask."""
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    position = 0
    for index, length in enumerate(lengths):
        if index % 2 == 1:  # odd runs are True
            flat[position : position + length] = True
        position += int(length)
    return flat.reshape(shape)


def save(path: str | Path, masks: dict[int, np.ndarray], shape: tuple[int, int]) -> None:
    """Write a frame-index -> mask mapping to one compressed file."""
    payload: dict[str, np.ndarray] = {
        "shape": np.asarray(shape, dtype=np.int64),
        "frames": np.asarray(sorted(masks), dtype=np.int64),
    }
    for frame_index, mask in masks.items():
        payload[f"m{frame_index}"] = encode(mask)
    np.savez_compressed(Path(path), **payload)


def load(path: str | Path) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    """Read back what ``save`` wrote."""
    with np.load(Path(path)) as data:
        shape = tuple(int(v) for v in data["shape"])
        frames = [int(v) for v in data["frames"]]
        masks = {i: decode(data[f"m{i}"], shape) for i in frames}
    return masks, shape
