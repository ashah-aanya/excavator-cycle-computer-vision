"""Choose the bucket seed frame by consensus of independent estimators. EXPERIMENT.

**Evaluation scaffolding, not pipeline code.** It lives in ``eval/`` so the dead-code gate
on ``src/`` does not apply while the idea is unproven. It reads no hand labels: the only
reference it scores against is a *good run's own bucket track* for the same video, used
to score and never to choose.

The problem
-----------
The bucket seed is one frame's geodesic band. On some runs that band is a sliver of stick
(the bucket was dusty or buried and was not in the excavator mask), SAM tracks the sliver
with full confidence, and the run collapses. Nothing checks the seed, and the scores of
the frames that compete for it are nearly tied.

The rule, written before any result was looked at
-------------------------------------------------
1. Candidates: anchor frames (the ~1 Hz frames the detector ran on) that have a mask and
   a band. Pool = the top third of the clip by arm extension AND mask-vs-detector
   agreement at or above the clip's median. (Extension is the straight-line distance from
   the pivot to the farthest mask pixel; agreement is the pipeline's own
   ``detection_mask_iou``.) If the pool is empty, fall back to the top third by extension.
2. For each candidate, three estimators locate the bucket independently:
     geodesic    the outer 18% of the walk along the mask (the pipeline's band, unchanged)
     thickness   the bucket is much wider than the stick: the far end of the arm where the
                 local thickness exceeds the stick's own thickness
     appearance  GrabCut on the image around the tip of the arm, seeded only by the tip's
                 position: the connected blob whose colours differ from its surroundings
3. Frame score = the *weakest* pairwise agreement between the estimators that returned a
   region (at least two must). Agreement is the overlap coefficient (intersection over the
   smaller area), because the band covers part of a bucket and the others cover more; two
   regions whose areas differ by more than 6x do not agree.
4. Seed = the highest-scoring candidate (ties to the longer extension). The seed region =
   pixels that at least two estimators support.

Everything is a ratio, an overlap, or a fraction of the arm's own length, so nothing depends
on a video's pixel scale.

Run
---
    PYTHONPATH=src python eval/bucket_consensus.py runs.json --out results/

``runs.json`` is a list of ``{"label", "video", "track", "masks", "ref_track", "ref_masks"}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from excavator_cycles import masks as mask_io
from excavator_cycles.kinematics import body_core, boom_base, geodesic_distance
from excavator_cycles.seeding import geodesic_bands

ESTIMATORS = ("geodesic", "thickness", "appearance")
MIN_PIXELS = 25
MAX_AREA_RATIO = 6.0
THICK_BINS = 48
STICK = (0.50, 0.75)  # the stick's range along the arm, as in seeding.geodesic_bands
TIP_FRACTION = 0.95  # the tip = arm pixels this far along the walk
WINDOW = 0.35  # appearance window half-side, as a fraction of the pivot-to-tip length


# --------------------------------------------------------------------------
# The arm, once per frame
# --------------------------------------------------------------------------


def arm_geometry(mask, core, pivot):
    """Walking distance along the mask, the arm's reach, and the pixels that are arm."""
    dist = geodesic_distance(mask, pivot)
    reachable = dist >= 0
    if not reachable.any():
        return dist, 0.0, np.zeros_like(mask, dtype=bool)
    body = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return dist, float(dist[reachable].max()), reachable & ~body


def extension(mask, pivot):
    """Straight-line distance from the pivot to the farthest mask pixel."""
    ys, xs = np.nonzero(mask)
    return float(np.hypot(xs - pivot[0], ys - pivot[1]).max()) if len(xs) else 0.0


# --------------------------------------------------------------------------
# The three estimators. Each returns a boolean mask, or None when it cannot say.
# --------------------------------------------------------------------------


def estimate_geodesic(mask, core, pivot):
    """The pipeline's own band: the outer 18% of the walk, body excluded."""
    band = geodesic_bands(mask, core, pivot)[0]
    return band if band.sum() >= MIN_PIXELS else None


def estimate_thickness(mask, core, pivot):
    """The far end of the arm where it is markedly thicker than the stick.

    Thickness per stretch of the walk is twice the largest distance-transform value in the
    stretch. The stick's own thickness (median and spread over its range) sets the bar; the
    bucket starts where, walking in from the tip, the arm stops exceeding it. ``None`` when
    the far end is never thicker than the stick, which is what a bucket seen edge-on, or
    absent from the mask, looks like.
    """
    dist, reach, live = arm_geometry(mask, core, pivot)
    if reach <= 0 or not live.any():
        return None
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    edges = np.linspace(0.0, reach, THICK_BINS + 1)
    thick = np.full(THICK_BINS, np.nan)
    for k in range(THICK_BINS):
        last = k == THICK_BINS - 1
        chosen = (
            live
            & (dist >= edges[k])
            & ((dist <= edges[k + 1]) if last else (dist < edges[k + 1]))
        )
        if chosen.sum() >= 3:
            thick[k] = 2.0 * float(dt[chosen].max())
    centres = (edges[:-1] + edges[1:]) / 2.0 / reach
    reference = thick[(centres >= STICK[0]) & (centres < STICK[1]) & np.isfinite(thick)]
    if reference.size < 3:
        return None
    median = float(np.median(reference))
    spread = float(np.median(np.abs(reference - median)))
    bar = max(median + 3.0 * spread, 1.25 * median)
    outer = np.where((centres >= STICK[1]) & np.isfinite(thick) & (thick >= bar))[0]
    if outer.size == 0:
        return None
    start = int(outer[np.argmax(thick[outer])])  # the widest bin, the body of the bucket
    while start - 1 >= 0 and np.isfinite(thick[start - 1]) and thick[start - 1] >= bar:
        start -= 1
    region = live & (dist >= edges[start])
    if region.sum() < MIN_PIXELS or region.sum() > 0.6 * live.sum():
        return None
    return region


def arm_tip(mask, core, pivot):
    """(tip position, pivot-to-tip length) of the arm: the far end of the walk, or None."""
    dist, reach, live = arm_geometry(mask, core, pivot)
    if reach <= 0:
        return None
    end = live & (dist >= TIP_FRACTION * reach)
    if not end.any():
        return None
    ys, xs = np.nonzero(end)
    tip = np.array([xs.mean(), ys.mean()])
    return tip, float(np.hypot(*(tip - np.asarray(pivot))))


def estimate_appearance(image_bgr, mask, core, pivot):
    """GrabCut around the arm's tip, seeded only by where the tip is.

    Independent of the geodesic band: it uses the tip's position and the picture's own
    colours. Foreground seed = a small disc at the tip; everything else in the window is
    probable background, with a border strip as definite background. The result is the
    connected blob containing the tip.
    """
    found = arm_tip(mask, core, pivot)
    if found is None:
        return None
    tip, length = found
    half = int(max(20, WINDOW * length))
    height, width = mask.shape
    x0, x1 = max(int(tip[0]) - half, 0), min(int(tip[0]) + half, width)
    y0, y1 = max(int(tip[1]) - half, 0), min(int(tip[1]) + half, height)
    crop = np.ascontiguousarray(image_bgr[y0:y1, x0:x1])
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return None
    state = np.full(crop.shape[:2], cv2.GC_PR_BGD, np.uint8)
    state[:3, :] = state[-3:, :] = state[:, :3] = state[:, -3:] = cv2.GC_BGD
    cy, cx = int(tip[1]) - y0, int(tip[0]) - x0
    cv2.circle(state, (cx, cy), max(6, int(0.09 * length)), cv2.GC_PR_FGD, -1)
    cv2.circle(state, (cx, cy), max(3, int(0.03 * length)), cv2.GC_FGD, -1)
    try:
        cv2.grabCut(
            crop, state, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK
        )
    except cv2.error:
        return None
    blob = ((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels = cv2.connectedComponents(blob)
    if (
        count <= 1
        or labels[min(max(cy, 0), blob.shape[0] - 1), min(max(cx, 0), blob.shape[1] - 1)] == 0
    ):
        return None
    keep = (
        labels
        == labels[min(max(cy, 0), blob.shape[0] - 1), min(max(cx, 0), blob.shape[1] - 1)]
    )
    full = np.zeros(mask.shape, dtype=bool)
    full[y0:y1, x0:x1] = keep
    return full if full.sum() >= MIN_PIXELS else None


# --------------------------------------------------------------------------
# Agreement between estimators, and the frame's consensus
# --------------------------------------------------------------------------


def agreement(a, b):
    """Overlap coefficient (intersection over the smaller area); 0 if sizes are far apart."""
    area_a, area_b = int(a.sum()), int(b.sum())
    small, large = min(area_a, area_b), max(area_a, area_b)
    if small < MIN_PIXELS or large / small > MAX_AREA_RATIO:
        return 0.0
    return float(np.logical_and(a, b).sum() / small)


def frame_consensus(regions, require=()):
    """(score, supported region). Score = weakest pairwise agreement; needs two estimators.

    ``require`` names estimators that must return a region: a frame where one abstains
    scores 0. The thickness estimator abstains exactly when the arm has no thick end, which
    is what a mask that lacks the bucket looks like, so it is a witness that can say no.
    """
    present = {k: v for k, v in regions.items() if v is not None and v.sum() >= MIN_PIXELS}
    if len(present) < 2 or any(name not in present for name in require):
        return 0.0, None
    names = sorted(present)
    pairs = [
        agreement(present[a], present[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    ]
    votes = sum(v.astype(int) for v in present.values())
    return min(pairs), votes >= 2


# --------------------------------------------------------------------------
# Evaluation against a good run's bucket track
# --------------------------------------------------------------------------


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def rank01(values):
    order = np.argsort(np.argsort(values))
    return order / max(len(values) - 1, 1)


def decode_frames(video, wanted):
    import av

    found = {}
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in wanted:
                found[index] = frame.to_ndarray(format="bgr24")
            if index >= max(wanted):
                break
    return found


def candidates(track, masks, core, pivot):
    """Anchor frames with a mask and a band, with their extension and agreement."""
    rows = []
    for i, frame in enumerate(track["frames"]):
        agree = frame.get("detection_mask_iou")
        if agree is None or i not in masks or not masks[i].any():
            continue
        if (frame.get("detection_truck_iou") or 0.0) >= 0.30:
            continue
        if estimate_geodesic(masks[i], core, pivot) is None:
            continue
        rows.append((i, extension(masks[i], pivot), float(agree)))
    return rows


def choose_pool(rows):
    """Top third by extension AND agreement at or above the median; fall back to extension."""
    ext = np.array([r[1] for r in rows])
    agr = np.array([r[2] for r in rows])
    top = rank01(ext) >= 2 / 3
    pool = [r for r, t, a in zip(rows, top, agr >= np.median(agr), strict=True) if t and a]
    return pool or [r for r, t in zip(rows, top, strict=True) if t] or rows


def evaluate(run, methods=ESTIMATORS, require=()):
    track = json.loads(Path(run["track"]).read_text())
    ref_track = json.loads(Path(run["ref_track"]).read_text())
    masks = mask_io.load_objects(run["masks"])[0]["excavator"]
    ref = mask_io.load_objects(run["ref_masks"])[0]["bucket"]
    times = np.array([f["time_seconds"] for f in track["frames"]])
    ref_times = np.array([f["time_seconds"] for f in ref_track["frames"]])
    usable = [m for m in masks.values() if m.any()]
    core = body_core(usable)
    pivot = boom_base(core)
    rows = candidates(track, masks, core, pivot)
    pool = choose_pool(rows)

    def reference_at(i):
        j = int(np.argmin(np.abs(ref_times - times[i])))
        return ref.get(j)

    frames = decode_frames(
        run["video"], {track["frames"][i]["frame_index"] for i, _, _ in rows}
    )
    detail = []
    for i, ext, agr in rows:
        image = frames[track["frames"][i]["frame_index"]]
        regions = {
            "geodesic": estimate_geodesic(masks[i], core, pivot),
            "thickness": estimate_thickness(masks[i], core, pivot),
            "appearance": estimate_appearance(image, masks[i], core, pivot),
        }
        score, region = frame_consensus({k: regions[k] for k in methods}, require)
        target = reference_at(i)
        scored = {
            k: (
                iou(v, target)
                if v is not None and target is not None and target.any()
                else None
            )
            for k, v in regions.items()
        }
        consensus_iou = (
            iou(region, target)
            if region is not None and target is not None and target.any()
            else None
        )
        detail.append(
            dict(
                sample=i,
                time=float(times[i]),
                ext=ext,
                agr=agr,
                score=score,
                regions=regions,
                region=region,
                iou=scored,
                consensus_iou=consensus_iou,
                in_pool=any(i == p[0] for p in pool),
            )
        )
    return track, detail, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--methods", default=",".join(ESTIMATORS), help="estimators in the consensus"
    )
    parser.add_argument("--require", default="", help="estimators that must return a region")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for run in json.loads(args.runs.read_text()):
        methods = tuple(args.methods.split(","))
        require = tuple(x for x in args.require.split(",") if x)
        track, detail, _ = evaluate(run, methods, require)
        pool = [d for d in detail if d["in_pool"]]
        best = max(pool, key=lambda d: (d["score"], d["ext"]))
        by_ext = max(pool, key=lambda d: d["ext"])
        summary.append(
            dict(
                run=run["label"],
                anchors=len(detail),
                pool=len(pool),
                pipeline_sample=track["bucket_seed"]["sample_position"],
                consensus_pick=best["sample"],
                consensus_score=best["score"],
                consensus_iou=best["consensus_iou"],
                geodesic_iou_at_consensus=best["iou"]["geodesic"],
                extension_pick_geodesic_iou=by_ext["iou"]["geodesic"],
                standalone={
                    k: float(
                        np.mean(
                            [d["iou"][k] for d in detail if d["iou"][k] is not None]
                            or [np.nan]
                        )
                    )
                    for k in ("geodesic", "thickness", "appearance")
                },
                returned={
                    k: sum(d["regions"][k] is not None for d in detail)
                    for k in ("geodesic", "thickness", "appearance")
                },
            )
        )
        print(json.dumps(summary[-1], default=float))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
