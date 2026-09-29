"""Clear-view score: is the arm's tip in a sharp view, or lost in dust? EXPERIMENT.

**Evaluation scaffolding, not pipeline code.** It reads no hand labels; it scores against a
good run's own bucket track for the same video, used only to judge, never to choose.

Why
---
A bucket buried in a pile, or hidden in a dust plume, is not in the excavator mask, so the
geodesic band lands on the end of the stick and SAM tracks that with full confidence. The
frame choice (long extension, agreement at or above the median) cannot see this: extension
measures reach, and agreement compares the mask with the detector's box. This score uses
the picture itself, so it does not read the excavator mask's opinion of where the bucket is.

The rule, written before any result was looked at
-------------------------------------------------
Take a window of the image around the tip of the arm (half-side = 0.35 x the pivot-to-tip
length, the same window the appearance estimator used), convert it to grey, resize it to
64 x 64 so frames of different scale compare, and measure

  sharp   the variance of the Laplacian
  edges   the mean Sobel gradient magnitude

Each frame's score is its percentile among its own clip's anchor frames. Two questions:

  1. Does the score separate frames where the band lands on the real bucket
     (overlap >= 0.5 with the good run's bucket) from frames where it does not?  (AUC)
  2. If it picks the frame, is the band there better than picking by extension alone?
       S0  the longest extension (the baseline)
       S1  within the top third by extension, the highest score
       S2  the highest sum of the extension percentile and the score percentile
     Both measures are run for S1 and S2. Nothing is chosen after seeing the outcome.

Run
---
    PYTHONPATH=src python eval/clear_view.py runs.json

``runs.json`` is the same list ``bucket_consensus.py`` takes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from bucket_consensus import (
    WINDOW,
    arm_tip,
    decode_frames,
    estimate_geodesic,
    extension,
    iou,
    mask_io,
    rank01,
)

from excavator_cycles.kinematics import body_core, boom_base

SIZE = 64
MEASURES = ("sharp", "edges")


def tip_window(image_bgr, mask, core, pivot):
    """The grey image around the arm's tip, resized to SIZE x SIZE, or None."""
    found = arm_tip(mask, core, pivot)
    if found is None:
        return None
    tip, length = found
    half = int(max(20, WINDOW * length))
    height, width = mask.shape
    x0, x1 = max(int(tip[0]) - half, 0), min(int(tip[0]) + half, width)
    y0, y1 = max(int(tip[1]) - half, 0), min(int(tip[1]) + half, height)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return None
    grey = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def clear_view(image_bgr, mask, core, pivot):
    """{"sharp": ..., "edges": ...} for the window around the tip, or None."""
    window = tip_window(image_bgr, mask, core, pivot)
    if window is None:
        return None
    gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "sharp": float(cv2.Laplacian(window, cv2.CV_64F).var()),
        "edges": float(np.hypot(gx, gy).mean()),
    }


def auc(good, bad):
    """Chance that a random good frame scores above a random bad one (0.5 = no signal)."""
    if not len(good) or not len(bad):
        return None
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    return wins / (len(good) * len(bad))


def evaluate(run):
    track = json.loads(Path(run["track"]).read_text())
    ref_track = json.loads(Path(run["ref_track"]).read_text())
    masks = mask_io.load_objects(run["masks"])[0]["excavator"]
    ref = mask_io.load_objects(run["ref_masks"])[0]["bucket"]
    times = np.array([f["time_seconds"] for f in track["frames"]])
    ref_times = np.array([f["time_seconds"] for f in ref_track["frames"]])
    core = body_core([m for m in masks.values() if m.any()])
    pivot = boom_base(core)
    rows = []
    for i, frame in enumerate(track["frames"]):
        if frame.get("detection_mask_iou") is None or i not in masks or not masks[i].any():
            continue
        if (frame.get("detection_truck_iou") or 0.0) >= 0.30:
            continue
        band = estimate_geodesic(masks[i], core, pivot)
        if band is not None:
            rows.append((i, band))
    images = decode_frames(run["video"], {track["frames"][i]["frame_index"] for i, _ in rows})
    out = []
    for i, band in rows:
        view = clear_view(images[track["frames"][i]["frame_index"]], masks[i], core, pivot)
        target = ref.get(int(np.argmin(np.abs(ref_times - times[i]))))
        overlap = iou(band, target) if target is not None and target.any() else 0.0
        if view is not None:
            out.append(
                dict(
                    sample=i,
                    time=float(times[i]),
                    ext=extension(masks[i], pivot),
                    iou=overlap,
                    **view,
                )
            )
    return out


def selections(frames):
    """Band overlap at the frame each rule picks."""
    ext = np.array([f["ext"] for f in frames])
    picks = {"S0 longest extension": int(np.argmax(ext))}
    top = np.where(rank01(ext) >= 2 / 3)[0]
    for measure in MEASURES:
        score = np.array([f[measure] for f in frames])
        picks[f"S1 top-third extension, best {measure}"] = int(top[np.argmax(score[top])])
        picks[f"S2 extension + {measure} (rank sum)"] = int(
            np.argmax(rank01(ext) + rank01(score))
        )
    return {name: frames[k]["iou"] for name, k in picks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    table, aucs = {}, {m: [] for m in MEASURES}
    for run in json.loads(args.runs.read_text()):
        frames = evaluate(run)
        good = [f for f in frames if f["iou"] >= 0.5]
        bad = [f for f in frames if f["iou"] < 0.5]
        line = {m: auc([f[m] for f in good], [f[m] for f in bad]) for m in MEASURES}
        for m in MEASURES:
            aucs[m].append(line[m])
        for name, value in selections(frames).items():
            table.setdefault(name, []).append(value)
        shown = ", ".join("n/a" if v is None else f"{v:.2f}" for v in line.values())
        counts = f"frames {len(frames):3d} good {len(good):3d} bad {len(bad):3d}"
        print(f"{run['label']:32s} {counts} | AUC (sharp, edges) {shown}")
    print("\nband overlap at the chosen frame, per run (same order):")
    for name, values in table.items():
        print(
            f"  {name:42s} "
            + " ".join(f"{v:5.2f}" for v in values)
            + f"   mean {np.mean(values):.2f}"
        )


if __name__ == "__main__":
    main()
