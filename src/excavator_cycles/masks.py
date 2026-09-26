"""Storing masks compactly.

A boolean mask per sample is small in principle and large in practice: 870
samples of 480x272 is 113 MB raw, and that is the file every later stage has to
load. Run-length encoding takes it to a few MB, because a mask is mostly long
runs of False with a machine-shaped island of True in the middle.

Why not ``pycocotools``: it is a C extension, and this is twenty lines of NumPy
that needs no build step on a cluster node.

One file, several objects
-------------------------
Tracking gained a second object (the bucket), so one run now produces two mask
sets. They share a single file because ``np.savez_compressed`` **truncates**:
calling ``save`` twice on the same path deletes the first set rather than adding
to it. Each set therefore gets its own key prefix and its own index array, and
readers treat every set but the excavator's as optional -- which is what keeps
the ``masks.npz`` files written before the bucket existed loading unchanged.
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


# Which key prefix and index array each tracked object's masks live under.
# ``m``/``frames`` is the original layout, so a file written before the bucket
# existed is simply this table with its second row missing.
OBJECT_SETS: dict[str, tuple[str, str]] = {
    "excavator": ("m", "frames"),
    "bucket": ("b", "bucket_frames"),
}


def save_objects(
    path: str | Path,
    objects: dict[str, dict[int, np.ndarray]],
    shape: tuple[int, int],
) -> None:
    """Write every tracked object's masks to one compressed file.

    ``objects`` maps an object name from ``OBJECT_SETS`` to a *sample ordinal*
    -> mask mapping. Sample ordinal, never source frame index: the pipeline
    indexes everything downstream by position in the sampled sequence, and
    mixing the two has already produced one confident wrong answer on this
    project.

    A second ``save`` call cannot stand in for a second object -- NumPy rewrites
    the archive from scratch each time -- which is the whole reason this
    function takes every object at once.
    """
    payload: dict[str, np.ndarray] = {"shape": np.asarray(shape, dtype=np.int64)}
    for name, masks in objects.items():
        if name not in OBJECT_SETS:
            raise ValueError(
                f"unknown mask set {name!r}; expected one of {sorted(OBJECT_SETS)}"
            )
        prefix, index_key = OBJECT_SETS[name]
        payload[index_key] = np.asarray(sorted(masks), dtype=np.int64)
        for sample, mask in masks.items():
            payload[f"{prefix}{sample}"] = encode(mask)
    np.savez_compressed(Path(path), **payload)


def save(path: str | Path, masks: dict[int, np.ndarray], shape: tuple[int, int]) -> None:
    """Write the excavator masks alone, in the original single-object layout."""
    save_objects(path, {"excavator": masks}, shape)


def load_objects(
    path: str | Path,
) -> tuple[dict[str, dict[int, np.ndarray]], tuple[int, int]]:
    """Read back every mask set the file happens to contain.

    A set whose index array is absent is simply not in the result -- that is how
    a file from before the bucket was tracked reports "no bucket" rather than
    failing to load at all.
    """
    objects: dict[str, dict[int, np.ndarray]] = {}
    with np.load(Path(path)) as data:
        shape = tuple(int(v) for v in data["shape"])
        for name, (prefix, index_key) in OBJECT_SETS.items():
            if index_key not in data.files:
                continue
            samples = [int(v) for v in data[index_key]]
            objects[name] = {i: decode(data[f"{prefix}{i}"], shape) for i in samples}
    return objects, shape


def load(path: str | Path) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    """Read back the excavator masks. Unchanged signature, on purpose.

    Every stage after perception calls this, and several completed runs already
    sit on disk, so growing a second object must not change what this returns.
    """
    objects, shape = load_objects(path)
    return objects.get("excavator", {}), shape
