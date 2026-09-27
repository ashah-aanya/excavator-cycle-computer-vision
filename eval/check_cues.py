"""Check a candidate cue against hand-labelled phase onsets.

**Evaluation scaffolding, not pipeline code.** Like ``eval/score.py``, it lives
outside ``src/`` and imports nothing from ``excavator_cycles``, so the pipeline can
never reach the labels through it. It needs NumPy and SciPy, not the pipeline.

What it answers
---------------
A cue is a *shape* in one feature: "height: drop ends -> flat", "dx/dt: peak". The
state machine only looks for a phase after the previous phase has started, so a
shape that also occurs earlier in the cycle does no harm. The question is:

    Scanning forward from the previous phase's labelled start, is the FIRST
    stretch showing this shape the true start of this phase?

For every labelled onset of the phase it reports that first stretch, whether it
contains the true start (within the grading tolerance), and how far the stretch's
start and centre are from it.

How a shape is read
-------------------
At every sample, fit a straight line to the ``SIDE`` seconds before it and to the
``SIDE`` seconds after it. Each side is rising, falling or flat, where flat means
the line moves less than ``FLAT`` of the feature's range (p95 - p5) over ``SIDE``
seconds. The pair names the shape: ``falling`` then ``flat`` is "drop ends ->
flat"; ``rising`` then ``falling`` is "peak". When both sides move the same way,
the slope ratio adds "speeds up" (after is 1.5x steeper) or "slows down".

One simplification to keep in mind: the scan starts from the previous phase's
LABELLED start. In the pipeline it would start from the DETECTED one, so an error
there carries into the next search.

Usage::

    uv run python eval/check_cues.py --phase dig --feature speed_2d --shape "drop ends -> flat"
    uv run python eval/check_cues.py --phase dumping --search
    uv run python eval/check_cues.py --clip dev --phase swinging --feature d2x_dt2 --shape peak
    uv run python eval/check_cues.py --list-features

Exit code: 0 when the first match contains the true start in every cycle, 1 when it
does not, 2 when the inputs cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
FIXTURES = EVAL.parent / "tests" / "fixtures"
CLIPS = {
    "long": (FIXTURES / "long_clip" / "features.npz", EVAL / "labels_long_clip.json"),
    "dev": (FIXTURES / "dev_clip" / "features.npz", EVAL / "labels.json"),
}

PHASES = ("digging", "hauling", "dumping", "swinging")
ALIASES = {
    "dig": "digging",
    "haul": "hauling",
    "dump": "dumping",
    "swing": "swinging",
    "return swing": "swinging",
}

SIDE = 2.0  # seconds fitted on each side of a sample
FLAT = 0.08  # |change over SIDE| below this fraction of the range reads as flat
STEEPER = 1.5  # slope ratio for "speeds up" / "slows down"
DERIVATIVE_WINDOW = 0.9  # seconds, the pipeline's features.derivative_window_seconds

SHAPES = {
    ("falling", "flat"): "drop ends -> flat",
    ("rising", "flat"): "rise ends -> flat",
    ("flat", "rising"): "flat -> starts rising",
    ("flat", "falling"): "flat -> starts falling",
    ("rising", "falling"): "peak",
    ("falling", "rising"): "dip",
    ("rising", "rising"): "keeps rising",
    ("falling", "falling"): "keeps falling",
    ("flat", "flat"): "flat",
}
DETAILS = ("speeds up", "slows down")

# Columns of features.npz that are one number per sample and worth scanning.
STORED = (
    "bucket_x",
    "bucket_y",
    "height",
    "dh_dt",
    "d2h_dt2",
    "dx_dt",
    "speed_x",
    "rel_cabin_x",
    "rel_cabin_y",
    "rel_truck_x",
    "rel_truck_y",
    "truck_overlap",
    "aspect_ratio",
    "radius",
)
DERIVED = ("dy_dt", "speed_2d", "d2x_dt2", "box_w", "box_h", "box_area")


class CheckError(Exception):
    """The inputs could not be read. ``main`` turns this into one line, exit 2."""


# ----------------------------------------------------------------------- inputs


def load_onsets(path: Path) -> list[tuple[str, float]]:
    """Every labelled onset as (phase, seconds), in time order.

    Reads both label layouts: ``eval/labels.json`` (one cycle, named boundaries)
    and ``eval/labels_long_clip.json`` (a list of onsets).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"could not read labels at {path}: {exc}") from None
    fps = float(data["video"]["fps"])
    if "onsets" in data:
        pairs = [(o["phase"], o["frame"] / fps) for o in data["onsets"]["list"]]
    else:
        b = data["boundaries"]
        pairs = [(p, b[f"{p}_begins"] / fps) for p in PHASES] + [
            ("digging", b["cycle_ends"] / fps)
        ]
    pairs.sort(key=lambda p: p[1])
    for phase, _ in pairs:
        if phase not in PHASES:
            raise CheckError(f"{path}: unknown phase {phase!r}")
    return pairs


def tolerance_of(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["scoring"]["tolerance_seconds"])


def derivative(
    values: np.ndarray, times: np.ndarray, window: float = DERIVATIVE_WINDOW
) -> np.ndarray:
    """Savitzky-Golay first derivative, sized exactly as the pipeline sizes it.

    Mirrors ``excavator_cycles.onsets.derivative`` (quadratic fit, window in
    seconds rounded to an odd sample count, at least 3) without importing it.
    Interior gaps are interpolated; ``tests/test_check_cues.py`` checks the two agree.
    """
    from scipy.signal import savgol_filter

    step = float(np.median(np.diff(times)))
    width = max(3, max(1, round(window / step)))
    width = width if width % 2 else width + 1
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if not ok.all():
        idx = np.arange(len(v))
        v = np.interp(idx, idx[ok], v[ok])
    return savgol_filter(v, width, 2, deriv=1, delta=step)


def load_features(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Time axis and every scannable feature, stored and derived."""
    try:
        z = np.load(path)
    except (OSError, ValueError) as exc:
        raise CheckError(f"could not read features at {path}: {exc}") from None
    t = z["time_seconds"]
    feats = {k: np.asarray(z[k], dtype=float) for k in STORED if k in z.files}
    if "truck_overlap" in feats:
        feats["truck_overlap"] = np.nan_to_num(
            feats["truck_overlap"]
        )  # unmeasured = no overlap
    dy = derivative(z["bucket_y"], t)
    feats["dy_dt"] = dy
    feats["speed_2d"] = np.hypot(z["dx_dt"], dy)
    feats["d2x_dt2"] = derivative(z["dx_dt"], t)
    # The bucket box's size, not only its ratio: when width and height change
    # together the ratio hides it. In pixels -- shapes read trends relative to the
    # signal's own spread, so the camera's distance does not matter.
    box = np.asarray(z["bucket_box"], dtype=float)
    feats["box_w"] = box[:, 2] - box[:, 0]
    feats["box_h"] = box[:, 3] - box[:, 1]
    feats["box_area"] = feats["box_w"] * feats["box_h"]
    return t, feats


# ------------------------------------------------------------------------ shape


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """The median of the slopes between every pair of points.

    A least-squares line squares each point's distance, so one spike pulls on it
    hardest of all. The median of pairwise slopes ignores a spike the way a median
    income ignores one billionaire: it stays right with up to ~29% bad points.
    """
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    i, j = np.triu_indices(len(x), k=1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def _slope(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "theilsen":
        return _theil_sen(x, y)
    return float(np.polyfit(x, y, 1)[0])


def side_changes(
    values: np.ndarray, times: np.ndarray, side: float = SIDE, method: str = "ols"
) -> tuple[np.ndarray, np.ndarray]:
    """Line-fit change over ``side`` seconds before and after each sample, as a
    fraction of the feature's p95 - p5 range. NaN where a side runs off the clip.

    ``method`` is "ols" (least squares, the default and what the pipeline uses) or
    "theilsen" (median of pairwise slopes, robust to spikes)."""
    step = float(np.median(np.diff(times)))
    n = round(side / step)
    spread = np.nanpercentile(values, 95) - np.nanpercentile(values, 5)
    spread = spread if spread > 0 else 1.0
    x = np.arange(n + 1) * step
    before = np.full(len(values), np.nan)
    after = np.full(len(values), np.nan)
    for i in range(len(values)):
        if i - n >= 0:
            before[i] = _slope(x, values[i - n : i + 1], method) * side / spread
        if i + n < len(values):
            after[i] = _slope(x, values[i : i + n + 1], method) * side / spread
    return before, after


def _word(change: float, flat: float) -> str | None:
    if not np.isfinite(change):
        return None
    return "flat" if abs(change) < flat else ("rising" if change > 0 else "falling")


def shape_at(before: float, after: float, flat: float = FLAT) -> tuple[str | None, str]:
    """(shape, detail) for one sample. detail is "" unless a keep-going shape
    clearly speeds up or slows down."""
    b, a = _word(before, flat), _word(after, flat)
    if b is None or a is None:
        return None, ""
    detail = ""
    if b == a and b != "flat":
        ratio = abs(after) / abs(before)
        detail = (
            "speeds up" if ratio > STEEPER else ("slows down" if ratio < 1 / STEEPER else "")
        )
    return SHAPES[(b, a)], detail


def shapes(
    values: np.ndarray,
    times: np.ndarray,
    flat: float = FLAT,
    side: float = SIDE,
    method: str = "ols",
):
    before, after = side_changes(values, times, side, method)
    return [shape_at(b, a, flat) for b, a in zip(before, after, strict=True)]


def context_words(
    values: np.ndarray,
    times: np.ndarray,
    length: float,
    flat: float = FLAT,
    method: str = "ols",
    after: bool = False,
) -> list[str | None]:
    """The big trend over the ``length`` seconds BEFORE each sample (or AFTER it,
    with ``after=True``): one line fit, read as rising / falling / flat by the same
    rule as a shape side."""
    b, a = side_changes(values, times, length, method)
    return [_word(v, flat) for v in (a if after else b)]


def persist(hit: np.ndarray, times: np.ndarray, seconds: float) -> np.ndarray:
    """True where ``hit`` stays true for the next ``seconds`` without a break.

    The cue then fires where a stretch of at least that duration STARTS, so the
    onset time is unchanged -- only short-lived blips are refused."""
    if seconds <= 0:
        return hit.copy()
    out = np.zeros(len(hit), bool)
    for i in range(len(hit)):
        j = int(np.searchsorted(times, times[i] + seconds - 1e-9, side="right"))
        if j <= len(hit) and times[min(j, len(hit)) - 1] >= times[i] + seconds - 1e-6:
            out[i] = bool(hit[i:j].all())
    return out


# ------------------------------------------------------------------ first match


@dataclass(frozen=True)
class Match:
    """What the scan found for one labelled onset."""

    truth: float
    anchor: float
    stretch: tuple[float, float] | None
    tolerance: float

    @property
    def contains(self) -> bool:
        if self.stretch is None:
            return False
        lo, hi = self.stretch
        return lo - self.tolerance <= self.truth <= hi + self.tolerance

    @property
    def start_error(self) -> float | None:
        return None if self.stretch is None else self.stretch[0] - self.truth

    @property
    def centre_error(self) -> float | None:
        return None if self.stretch is None else sum(self.stretch) / 2 - self.truth


def first_matches(
    times: np.ndarray,
    labelled: list[tuple[str, float]],
    phase: str,
    hit: np.ndarray,
    tolerance: float,
    anchor_back: int = 1,
) -> list[Match]:
    """For each labelled onset of ``phase``, the first stretch where ``hit`` is
    true, scanning forward from the onset ``anchor_back`` places earlier."""
    out = []
    for k, (p, truth) in enumerate(labelled):
        if p != phase or k - anchor_back < 0:
            continue
        anchor = labelled[k - anchor_back][1]
        i = int(np.searchsorted(times, anchor))
        while i < len(times) and not hit[i]:
            i += 1
        if i == len(times):
            out.append(Match(truth, anchor, None, tolerance))
            continue
        j = i
        while j + 1 < len(times) and hit[j + 1]:
            j += 1
        out.append(Match(truth, anchor, (float(times[i]), float(times[j])), tolerance))
    return out


def hits(read: list[tuple[str | None, str]], shape: str, detail: str = "") -> np.ndarray:
    return np.array([s == shape and (not detail or d == detail) for s, d in read])


# ------------------------------------------------------------------ big drops


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


def drop_hits(values: np.ndarray, times: np.ndarray, **kw) -> np.ndarray:
    """True at the onset sample of every big drop -- a cue that fires once per fall."""
    hit = np.zeros(len(times), bool)
    for d in big_drops(values, times, **kw):
        hit[int(np.searchsorted(times, d["onset"]))] = True
    return hit


# ---------------------------------------------------------------- pattern windows
#
# Window finders: each returns the SPANS where a pattern is happening, as
# (start, end) in seconds. A phase's window is the first such span after the
# previous phase's window; it is judged by whether it contains the true start and
# by how wide it is. Nothing here picks an exact moment.


def _runs(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [
        (float(times[a]), float(times[b - 1]))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def knee_mask(
    values: np.ndarray,
    times: np.ndarray,
    direction: str = "rising",
    turn: str = "flat_to_steep",
    side: float = SIDE,
    flat_max: float = FLAT,
    steep_min: float = 0.25,
) -> np.ndarray:
    """Spans where the slope turns sharply, found from a line fitted to the ``side``
    seconds before each sample and another to the ``side`` seconds after.

    ``turn="flat_to_steep"``: before is flat (moves < ``flat_max`` of the spread),
    after is steep (moves >= ``steep_min`` of it) in ``direction``. A take-off.
    ``turn="steep_to_flat"``: the reverse. A landing -- "a plateau after a steep
    decrease" is ``direction="falling", turn="steep_to_flat"``."""
    before, after = side_changes(values, times, side)
    sign = 1.0 if direction == "rising" else -1.0
    flat_side, steep_side = (before, after) if turn == "flat_to_steep" else (after, before)
    with np.errstate(invalid="ignore"):
        return (np.abs(flat_side) < flat_max) & (steep_side * sign >= steep_min)


def knee_windows(values, times, direction="rising", turn="flat_to_steep", **kw):
    """`knee_mask` as spans."""
    return _runs(knee_mask(values, times, direction, turn, **kw), times)


def dip_then_rise_mask(
    values: np.ndarray,
    times: np.ndarray,
    side: float = SIDE,
    flat_max: float = FLAT,
    rise_min: float = 0.25,
) -> np.ndarray:
    """ "The bottom of a dip with a spike after": the line over the ``side`` s before
    is NOT rising (falling, or flat within ``flat_max`` of the spread) and the line
    over the ``side`` s after rises steeply (by at least ``rise_min`` of it). Unlike a
    take-off knee, a fall before is allowed -- the dip may be shallow or deep."""
    before, after = side_changes(values, times, side)
    with np.errstate(invalid="ignore"):
        return (before < flat_max) & (after >= rise_min)


def excursion_windows(
    values: np.ndarray,
    times: np.ndarray,
    direction: str = "dip",
    min_size: float = 0.5,
) -> list[dict]:
    """Every big dip (or peak): the signal moves at least ``min_size`` of its spread
    away from its level and back (prominence). Returns its start (where it leaves the
    level), apex, and end (where it stops coming back, i.e. has returned)."""
    v = np.asarray(values, dtype=float)
    spread = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
    spread = spread if spread > 0 else 1.0
    x = -v if direction == "dip" else v
    from scipy.signal import find_peaks

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


def settle_windows(
    values: np.ndarray,
    times: np.ndarray,
    direction: str = "dip",
    min_size: float = 0.5,
    recovered: float = 0.8,
    side: float = SIDE,
    flat: float = FLAT,
) -> list[dict]:
    """After each big dip (or peak): the stretch where the signal comes back and settles.

    The dip itself is CONTEXT -- it has to be seen -- but the output is to its right:
    from where the signal is ``recovered`` of the way back from the dip's bottom to
    the level it returns to, to where it has settled (the line over the next ``side``
    s is flat). Both ends are read from the signal, so no fixed shift is added.

    "The level it returns to", not the level it left: before the return swing dx/dt
    is positive (the bucket is pushed toward the truck first), so measuring against
    the level before the dip demanded a recovery that never comes.
    """
    v = np.asarray(values, dtype=float)
    x = -v if direction == "dip" else v
    _, after = side_changes(v, times, side)
    out = []
    for d in excursion_windows(values, times, direction, min_size):
        p = int(np.searchsorted(times, d["apex"]))
        z = int(np.searchsorted(times, d["end"]))
        back = x[z] + (1 - recovered) * (x[p] - x[z])
        r = p
        while r + 1 < len(x) and x[r] > back:
            r += 1
        e = r
        while e + 1 < len(x) and not (np.isfinite(after[e]) and abs(after[e]) < flat):
            e += 1
        out.append(
            {"context": (d["start"], d["end"]), "window": (float(times[r]), float(times[e]))}
        )
    return out


# ------------------------------------------------------------------------ rules
#
# Hard rules VETO a candidate moment: the phase cannot be starting there. Soft rules
# are evidence only, reported for the weighting layer to come. The classification
# is Aanya's (docs/stages/09-rules-draft.md, 2026-09-27).
#
# Memory within a cycle ("this cycle's dig height") uses only samples BEFORE the
# candidate moment, as the pipeline would have to.


def truck_side(feats: dict[str, np.ndarray]) -> float:
    """+1 if the truck is on the +x side of the cabin, -1 if not: the median sign of
    bucket - cabin x while the bucket overlaps the truck box. A simplification of the
    pipeline's `calibrate`, which thresholds the overlap first."""
    over = feats["truck_overlap"] > 0
    rel = feats["rel_cabin_x"][over & np.isfinite(feats["rel_cabin_x"])]
    return 1.0 if rel.size == 0 or np.median(rel) >= 0 else -1.0


def _running_median(values: np.ndarray, start: int) -> np.ndarray:
    """At each sample i > start: the median of values[start:i] (the past only)."""
    out = np.full(len(values), np.nan)
    for i in range(start + 1, len(values)):
        out[i] = np.nanmedian(values[start:i])
    return out


def _dig_start_before(times: np.ndarray, labelled, k: int) -> int | None:
    """Sample index of the latest labelled digging onset before onset ``k``."""
    for p, s in reversed(labelled[:k]):
        if p == "digging":
            return int(np.searchsorted(times, s))
    return None


def rule_masks(times, feats, labelled, k: int, side: float) -> dict[str, np.ndarray]:
    """Every rule's truth at every sample, for the scan that looks for onset ``k``."""
    h = feats["height"]
    n = len(times)
    d0 = _dig_start_before(times, labelled, k)
    dig_h = _running_median(h, d0) if d0 is not None else np.full(n, np.nan)
    dig_x = _running_median(feats["bucket_x"], d0) if d0 is not None else np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        return {
            "D1": feats["truck_overlap"] <= 0,
            "D2": feats["rel_truck_x"] * side < 0,
            "D3": feats["rel_cabin_x"] * side < 0,
            "D4": feats["rel_cabin_y"] < 0,
            "D5": h < 0,
            "H1": h > dig_h,
            "H3": (feats["bucket_x"] - dig_x) * side > 0,
            "P1": feats["rel_cabin_x"] * side > 0,
            "P2": h > dig_h,
            "P3": feats["rel_cabin_y"] > 0,
            "P4": h > 0,
            "P5": feats["rel_truck_y"] > 0,
            "P6": feats["truck_overlap"] > 0,
        }


RULES = {
    "digging": {"hard": ("D1", "D2"), "soft": ("D3", "D4", "D5")},
    "hauling": {"hard": ("H1",), "soft": ("H3",)},
    "dumping": {"hard": ("P1", "P2"), "soft": ("P3", "P4", "P5", "P6")},
    "swinging": {"hard": (), "soft": ()},
}


def first_matches_ruled(
    times: np.ndarray,
    feats: dict[str, np.ndarray],
    labelled: list[tuple[str, float]],
    phase: str,
    hit: np.ndarray,
    tolerance: float,
    anchor_back: int = 1,
) -> list[tuple[Match, dict[str, bool]]]:
    """`first_matches` with the phase's HARD rules applied as a veto. Also returns,
    for each match, which SOFT rules held at the matched moment."""
    side = truck_side(feats)
    out = []
    for k, (p, _) in enumerate(labelled):
        if p != phase or k - anchor_back < 0:
            continue
        masks = rule_masks(times, feats, labelled, k, side)
        allowed = hit.copy()
        for rid in RULES[phase]["hard"]:
            allowed &= masks[rid]
        (m,) = first_matches(times, labelled[: k + 1], phase, allowed, tolerance, anchor_back)[
            -1:
        ]
        soft = {}
        if m.stretch is not None:
            i = int(np.searchsorted(times, m.stretch[0]))
            soft = {rid: bool(masks[rid][i]) for rid in RULES[phase]["soft"]}
        out.append((m, soft))
    return out


# ---------------------------------------------------------------------- reports


def _fmt(v: float | None) -> str:
    return "   --" if v is None else f"{v:+5.1f}"


def report(
    feature: str, shape: str, detail: str, matches: list[Match], anchor_name: str
) -> str:
    name = shape + (f", {detail}" if detail else "")
    lines = [f"{feature}: {name}   (scanning forward from {anchor_name}'s labelled start)", ""]
    lines.append("  labelled   first match        contains?  start err  centre err")
    for m in matches:
        where = (
            "no match" if m.stretch is None else f"{m.stretch[0]:6.1f} - {m.stretch[1]:5.1f} s"
        )
        mark = "yes" if m.contains else "NO"
        lines.append(
            f"  {m.truth:7.2f} s  {where:<17}  {mark:<9}  "
            f"{_fmt(m.start_error)} s    {_fmt(m.centre_error)} s"
        )
    n_ok = sum(m.contains for m in matches)
    lines += ["", f"  first match contains the start in {n_ok}/{len(matches)} cycles"]
    return "\n".join(lines)


def search(
    times: np.ndarray,
    feats: dict[str, np.ndarray],
    labelled: list[tuple[str, float]],
    phase: str,
    tolerance: float,
    anchor_back: int = 1,
) -> list[dict]:
    """Every feature x shape (x detail), scored and ranked."""
    rows = []
    for name, values in feats.items():
        read = shapes(values, times)
        for shape in SHAPES.values():
            options = [""] + (list(DETAILS) if shape.startswith("keeps") else [])
            for detail in options:
                ms = first_matches(
                    times, labelled, phase, hits(read, shape, detail), tolerance, anchor_back
                )
                if not ms:
                    continue
                starts = [abs(m.start_error) for m in ms if m.start_error is not None]
                centres = [abs(m.centre_error) for m in ms if m.centre_error is not None]
                rows.append(
                    {
                        "feature": name,
                        "shape": shape,
                        "detail": detail,
                        "contains": sum(m.contains for m in ms),
                        "n": len(ms),
                        "start_within": sum(e <= tolerance for e in starts),
                        "centre_within": sum(e <= tolerance for e in centres),
                        "worst_start": max(starts) if len(starts) == len(ms) else float("inf"),
                        "worst_centre": max(centres)
                        if len(centres) == len(ms)
                        else float("inf"),
                    }
                )
    rows.sort(
        key=lambda r: (
            -r["contains"],
            -max(r["start_within"], r["centre_within"]),
            min(r["worst_start"], r["worst_centre"]),
        )
    )
    return rows


CONTEXT_LENGTHS = (3.0, 4.0, 5.0, 6.0, 8.0)
TIMING_SIDES = (1.0, 1.5, 2.0)


def context_search(clips: dict, phase: str, anchor_back: int = 1) -> list[dict]:
    """Every feature x context (length, word) x timing (side, shape), scored on
    every clip given. ``clips`` maps a name to (times, feats, labelled, tolerance).

    A cue here is: "over the last L seconds the feature was <word>, AND its short
    shape (side S) is <shape> now". The long window says WHETHER this is the moment;
    the short one says exactly WHEN.
    """
    names = sorted(set.intersection(*(set(c[1]) for c in clips.values())))
    rows = []
    for name in names:
        per_clip = {}
        for clip, (t, feats, _, _) in clips.items():
            v = feats[name]
            per_clip[clip] = (
                {L: np.array(context_words(v, t, L), dtype=object) for L in CONTEXT_LENGTHS},
                {S: shapes(v, t, side=S) for S in TIMING_SIDES},
            )
        for L in CONTEXT_LENGTHS:
            for word in ("rising", "falling", "flat"):
                for S in TIMING_SIDES:
                    for shape in SHAPES.values():
                        row = {
                            "feature": name,
                            "context": f"{word} over {L:g} s",
                            "timing": f"{shape} ({S:g} s)",
                        }
                        for clip, (t, _, labelled, tol) in clips.items():
                            ctx, short = per_clip[clip]
                            hit = (ctx[L] == word) & hits(short[S], shape)
                            ms = first_matches(t, labelled, phase, hit, tol, anchor_back)
                            errs = [
                                abs(m.start_error) for m in ms if m.start_error is not None
                            ]
                            row[clip] = {
                                "contains": sum(m.contains for m in ms),
                                "timed": sum(
                                    m.start_error is not None and abs(m.start_error) <= tol
                                    for m in ms
                                ),
                                "n": len(ms),
                                "worst": max(errs)
                                if len(errs) == len(ms) and ms
                                else float("inf"),
                                "starts": [m.start_error for m in ms],
                            }
                        rows.append(row)
    first = next(iter(clips))
    rows.sort(
        key=lambda r: (
            -r[first]["timed"],
            -sum(r[c]["timed"] for c in clips),
            r[first]["worst"],
        )
    )
    return rows


# ------------------------------------------------------------------------- main


def _phase(name: str) -> str:
    name = ALIASES.get(name.lower(), name.lower())
    if name not in PHASES:
        raise CheckError(f"unknown phase {name!r}; expected one of {', '.join(PHASES)}")
    return name


def _shape(name: str) -> str:
    name = name.replace("→", "->").strip()
    if name not in SHAPES.values():
        raise CheckError(
            f"unknown shape {name!r}; expected one of: {', '.join(SHAPES.values())}"
        )
    return name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", choices=sorted(CLIPS), default="long")
    ap.add_argument("--features", type=Path, help="features.npz (overrides --clip)")
    ap.add_argument("--labels", type=Path, help="labels JSON (overrides --clip)")
    ap.add_argument("--phase", help="digging, hauling, dumping or swinging")
    ap.add_argument("--feature", help="a feature name; see --list-features")
    ap.add_argument("--shape", help=f"one of: {', '.join(SHAPES.values())}")
    ap.add_argument("--detail", default="", choices=("", *DETAILS))
    ap.add_argument(
        "--anchor-back",
        type=int,
        default=1,
        help="scan from the onset this many places back (1 = previous phase)",
    )
    ap.add_argument(
        "--search", action="store_true", help="rank every feature x shape for --phase"
    )
    ap.add_argument(
        "--context-search",
        action="store_true",
        help="rank long context x short timing cues for --phase, on both clips",
    )
    ap.add_argument(
        "--rules",
        action="store_true",
        help="apply the phase's hard rules as a veto; report soft rules",
    )
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--list-features", action="store_true")
    args = ap.parse_args(argv)

    try:
        feat_path = args.features or CLIPS[args.clip][0]
        label_path = args.labels or CLIPS[args.clip][1]
        times, feats = load_features(feat_path)
        if args.list_features:
            print("\n".join(f"{k}{'  (derived)' if k in DERIVED else ''}" for k in feats))
            return 0
        if not args.phase:
            raise CheckError("--phase is required")
        phase = _phase(args.phase)
        labelled = load_onsets(label_path)
        tol = tolerance_of(label_path)
        anchor_name = PHASES[(PHASES.index(phase) - args.anchor_back) % 4]

        if args.context_search:
            clips = {}
            for name in ("long", "dev"):
                fp, lp = CLIPS[name]
                t, feats = load_features(fp)
                clips[name] = (t, feats, load_onsets(lp), tolerance_of(lp))
            rows = context_search(clips, phase, args.anchor_back)
            print(
                f"{phase}: long context x short timing, "
                f"first match after {anchor_name} starts."
                "\n'timed' = the matching stretch STARTS within the tolerance of the label."
                "\nRanked on the 83 s clip; the dev clip was not used to choose anything.\n"
            )
            print(
                f"  {'feature':<14} {'context':<18} {'timing':<32} 83 s timed   dev timed   "
                "83 s start errors (s)"
            )
            for r in rows[: args.top]:
                lo, dv = r["long"], r["dev"]
                errs = " ".join("  -- " if e is None else f"{e:+.1f}" for e in lo["starts"])
                print(
                    f"  {r['feature']:<14} {r['context']:<18} {r['timing']:<32} "
                    f"{lo['timed']}/{lo['n']}          {dv['timed']}/{dv['n']}         {errs}"
                )
            return 0

        if args.search:
            rows = search(times, feats, labelled, phase, tol, args.anchor_back)
            print(
                f"{phase}: every feature x shape, first match after {anchor_name} starts "
                f"({feat_path.parent.name}, tolerance {tol:g} s)\n"
            )
            print(
                f"  {'feature':<14} {'shape':<34} contains  start<=tol  centre<=tol"
                "  worst start  worst centre"
            )
            for r in rows[: args.top]:
                name = r["shape"] + (f", {r['detail']}" if r["detail"] else "")
                print(
                    f"  {r['feature']:<14} {name:<34} {r['contains']}/{r['n']}       "
                    f"{r['start_within']}/{r['n']}         {r['centre_within']}/{r['n']}"
                    f"          {r['worst_start']:5.1f} s      {r['worst_centre']:5.1f} s"
                )
            best = rows[0] if rows else None
            return 0 if best and best["contains"] == best["n"] else 1

        if not args.feature or not args.shape:
            raise CheckError("give --feature and --shape, or --search")
        if args.feature not in feats:
            raise CheckError(f"unknown feature {args.feature!r}; see --list-features")
        shape = _shape(args.shape)
        read = shapes(feats[args.feature], times)
        if args.rules:
            ruled = first_matches_ruled(
                times,
                feats,
                labelled,
                phase,
                hits(read, shape, args.detail),
                tol,
                args.anchor_back,
            )
            ms = [m for m, _ in ruled]
            print(report(args.feature, shape, args.detail, ms, anchor_name))
            hard = ", ".join(RULES[phase]["hard"]) or "none"
            print(f"\n  hard rules applied as a veto: {hard}")
            for m, soft in ruled:
                held = ", ".join(f"{r} {'yes' if v else 'NO'}" for r, v in soft.items())
                print(f"  soft rules at {m.truth:.2f} s's match: {held or 'none'}")
            return 0 if ms and all(m.contains for m in ms) else 1
        ms = first_matches(
            times, labelled, phase, hits(read, shape, args.detail), tol, args.anchor_back
        )
        print(report(args.feature, shape, args.detail, ms, anchor_name))
        return 0 if ms and all(m.contains for m in ms) else 1
    except CheckError as exc:
        print(f"check_cues: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
