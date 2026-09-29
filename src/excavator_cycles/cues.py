"""The signals and pattern finders the phase windows are built from.

`windows.py` asks questions like "where is the big drop in height?" or "where does
the bucket's position along the pile-to-truck line turn from flat to steep?". This
module answers them: it derives the extra signals the questions need from the
pipeline's feature table, and it finds the patterns.

Everything here is read relative to the signal it is looking at -- a fraction of the
signal's own p95 - p5 spread, or a multiple of its own noise -- so the same code
serves any camera distance or resolution. The durations (2 s either side of a
sample, and so on) are fixed seconds, chosen while looking at two labelled clips.
The report lists them as such.
"""

from __future__ import annotations

import numpy as np

from .features import FeatureTable, Scene
from .rates import derivative

SIDE = 2.0  # seconds fitted on each side of a sample
FLAT = 0.08  # |change over SIDE| below this fraction of the spread reads as flat
PILE_QUANTILE = 0.10  # the lowest tenth of bucket heights: where it digs


class NoPhases(Exception):
    """The signals cannot support a phase search at all (no truck, no pile).

    Not a failure of the pipeline: some videos simply do not show what the method
    needs. `cycles` turns it into an answer with no cycles and says why.
    """


# ---------------------------------------------------------------------- signals


def over_truck(bucket_box: np.ndarray, truck_box: np.ndarray) -> np.ndarray:
    """How much of the bucket is OVER the truck, in [0, 1].

    Two boxes touching in the picture is not the bucket being over the truck: seen
    from in front, a bucket digging in the ground in front of the truck overlaps
    the truck's box. A truck's bed sits on its wheels, which fill the lower half of
    its box, so a bucket over the bed has its centre ABOVE the box's middle (image
    y down). Over = the share of the bucket's width inside the truck's left-right
    span, when the bucket's centre is above the truck box's middle; else 0. The
    bucket may be above the box's top edge and still be over the truck."""
    x1, y1, x2, y2 = (bucket_box[:, i] for i in range(4))
    tx1, ty1, tx2, ty2 = truck_box
    width = x2 - x1
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = np.clip(np.minimum(x2, tx2) - np.maximum(x1, tx1), 0, None) / width
        above_middle = (y1 + y2) / 2 < (ty1 + ty2) / 2
    return np.nan_to_num(np.where(above_middle, inside, 0.0))


def pile_truck_axis(feats: dict[str, np.ndarray], t: np.ndarray) -> dict[str, np.ndarray]:
    """Where the bucket is along the line from the PILE to the TRUCK: -1 at the pile,
    0 at the truck's centre, + past the truck on the far side.

    Straight-line distance loses which side of the truck the bucket is on; this keeps
    it without assuming a side-on camera. The pile is found from the video (the
    bucket's median position over its lowest tenth of heights, off the truck), so the
    line can point in any image direction. Image vectors with y DOWN, in L.
    """
    bucket = np.stack([feats["rel_truck_x"], -feats["rel_truck_y"]], axis=1)  # - truck
    low = feats["height"] <= np.nanquantile(feats["height"], PILE_QUANTILE)
    low &= ~(feats["truck_overlap"] > 0)
    if not (low & np.isfinite(bucket).all(axis=1)).any():
        raise NoPhases(
            "no pile position was found: the bucket is never low while off the truck"
        )
    pile = np.nanmedian(bucket[low], axis=0)  # the pile, relative to the truck
    norm2 = float(pile @ pile)
    pos = -(bucket @ pile) / norm2
    # the cabin on the same line: (bucket - truck) - (bucket - cabin)
    cabin = bucket - np.stack([feats["rel_cabin_x"], -feats["rel_cabin_y"]], axis=1)
    return {
        "pile_truck_pos": pos,
        "pile_truck_pos_dt": derivative(pos, t),
        "cabin_pile_truck_pos": -(cabin @ pile) / norm2,
    }


def signals(table: FeatureTable, scene: Scene) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The time axis and every signal the phase windows read, from the feature table."""
    t = np.asarray(table.time_seconds, dtype=float)
    feats = {
        name: np.asarray(getattr(table, name), dtype=float)
        for name in (
            "bucket_x",
            "height",
            "dh_dt",
            "dx_dt",
            "rel_cabin_x",
            "rel_cabin_y",
            "rel_truck_x",
            "rel_truck_y",
            "aspect_ratio",
        )
    }
    # unmeasured = no overlap. When the truck's box is known, "overlap" means OVER the
    # truck (see over_truck), not two boxes touching in the picture.
    feats["truck_overlap"] = np.nan_to_num(np.asarray(table.truck_overlap, dtype=float))
    if scene.truck_box is not None:
        feats["truck_overlap"] = over_truck(
            np.asarray(table.bucket_box, dtype=float), np.asarray(scene.truck_box, dtype=float)
        )
    feats["speed_2d"] = np.hypot(table.dx_dt, derivative(table.bucket_y, t))
    feats["aspect_ratio_dt"] = derivative(feats["aspect_ratio"], t)
    if not np.isfinite(feats["rel_truck_x"]).any():
        raise NoPhases(
            "no truck was detected: every rule measures the bucket against the dump "
            "truck, so no phase windows can be found"
        )
    feats.update(pile_truck_axis(feats, t))
    return t, feats


def truck_side(feats: dict[str, np.ndarray]) -> float:
    """+1 if the truck is on the +x side of the cabin, -1 if not: the median sign of
    bucket - cabin x while the bucket is over the truck."""
    over = feats["truck_overlap"] > 0
    rel = feats["rel_cabin_x"][over & np.isfinite(feats["rel_cabin_x"])]
    return 1.0 if rel.size == 0 or np.median(rel) >= 0 else -1.0


def running_median(values: np.ndarray, start: int) -> np.ndarray:
    """At each sample i > start: the median of values[start:i] (the past only)."""
    out = np.full(len(values), np.nan)
    for i in range(start + 1, len(values)):
        out[i] = np.nanmedian(values[start:i])
    return out


# --------------------------------------------------------------------- patterns


def side_changes(
    values: np.ndarray, times: np.ndarray, side: float = SIDE
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line-fit change over ``side`` seconds before and after each
    sample, as a fraction of the feature's p95 - p5 spread. NaN where a side runs off
    the clip."""
    step = float(np.median(np.diff(times)))
    n = round(side / step)
    spread = np.nanpercentile(values, 95) - np.nanpercentile(values, 5)
    spread = spread if spread > 0 else 1.0
    x = np.arange(n + 1) * step
    before = np.full(len(values), np.nan)
    after = np.full(len(values), np.nan)
    for i in range(len(values)):
        if i - n >= 0:
            before[i] = float(np.polyfit(x, values[i - n : i + 1], 1)[0]) * side / spread
        if i + n < len(values):
            after[i] = float(np.polyfit(x, values[i : i + n + 1], 1)[0]) * side / spread
    return before, after


def runs(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    """The (start, end) times of each stretch where ``mask`` is true."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [
        (float(times[a]), float(times[b - 1]))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def big_drops(
    values: np.ndarray,
    times: np.ndarray,
    min_drop: float = 0.5,
    max_seconds: float = 2.5,
    rest: float = 0.1,
) -> list[dict]:
    """Every big, fast fall: the signal loses at least ``min_drop`` of its p95 - p5
    spread within ``max_seconds``.

    For each fall, walk out from its steepest point: BACK to where it left the level
    before (the onset) and FORWARD to where it bottoms out. "Still falling" means a
    slope steeper than ``rest`` spreads per second; anything gentler is the level.
    Returns dicts with onset, steepest and bottom times, and the fall as a fraction
    of the spread.
    """
    v = np.asarray(values, dtype=float)
    spread = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
    spread = spread if spread > 0 else 1.0
    slope = np.gradient(v, times) / spread  # spreads per second
    falling = slope < -rest
    out, seen = [], set()
    for s in range(1, len(v) - 1):
        if not (falling[s] and slope[s] <= slope[s - 1] and slope[s] <= slope[s + 1]):
            continue  # not the steepest point of a fall
        a = s
        while a > 0 and falling[a - 1]:
            a -= 1
        b = s
        while b + 1 < len(v) and falling[b + 1]:
            b += 1
        if a in seen:
            continue
        seen.add(a)
        drop = (v[a] - v[b]) / spread
        if drop >= min_drop and times[b] - times[a] <= max_seconds:
            out.append(
                {
                    "onset": float(times[a]),
                    "steepest": float(times[s]),
                    "bottom": float(times[b]),
                    "drop": float(drop),
                }
            )
    return out


def knee_mask(
    values: np.ndarray,
    times: np.ndarray,
    direction: str = "rising",
    turn: str = "flat_to_steep",
    side: float = SIDE,
    flat_max: float = FLAT,
    steep_min: float = 0.25,
) -> np.ndarray:
    """Where the slope turns sharply, from a line fitted to the ``side`` seconds
    before each sample and another to the ``side`` seconds after.

    ``turn="flat_to_steep"``: before is flat (moves < ``flat_max`` of the spread),
    after is steep (moves >= ``steep_min`` of it) in ``direction``. A take-off.
    ``turn="steep_to_flat"``: the reverse. A landing."""
    before, after = side_changes(values, times, side)
    sign = 1.0 if direction == "rising" else -1.0
    flat_side, steep_side = (before, after) if turn == "flat_to_steep" else (after, before)
    with np.errstate(invalid="ignore"):
        return (np.abs(flat_side) < flat_max) & (steep_side * sign >= steep_min)


def knee_windows(values, times, direction="rising", turn="flat_to_steep", **kw):
    """`knee_mask` as spans."""
    return runs(knee_mask(values, times, direction, turn, **kw), times)


def excursion_windows(
    values: np.ndarray,
    times: np.ndarray,
    direction: str = "dip",
    min_size: float = 0.5,
) -> list[dict]:
    """Every big dip (or peak): the signal moves at least ``min_size`` of its spread
    away from its level and back (prominence). Returns its start (where it leaves the
    level), apex, and end (where it stops coming back, i.e. has returned)."""
    from scipy.signal import find_peaks

    v = np.asarray(values, dtype=float)
    spread = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
    spread = spread if spread > 0 else 1.0
    x = -v if direction == "dip" else v
    peaks, _ = find_peaks(x, prominence=min_size * spread)
    out = []
    for p in peaks:
        a = p
        while a > 0 and x[a - 1] < x[a]:
            a -= 1  # walk down the leading flank to where it started
        b = p
        while b + 1 < len(x) and x[b + 1] < x[b]:
            b += 1  # walk down the trailing flank to where it settled
        out.append({"start": float(times[a]), "apex": float(times[p]), "end": float(times[b])})
    return out
