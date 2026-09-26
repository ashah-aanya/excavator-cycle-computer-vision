"""Annotated video showing BOXES rather than masks, with their provenance on screen.

Where each box comes from -- the point of the caption strip:

    machine box  = bbox of SAM 2 object 1, seeded once by a Grounding DINO
                   "excavator." box, propagated to all 296 samples
    bucket box   = bbox of SAM 2 object 2, seeded once by GEODESIC GEOMETRY
                   (farthest point along the metal), propagated the same way

No detector ever located the bucket. Grounding DINO boxed the whole machine on
every bucket prompt, and the YOLO-World probe landed on the dump truck's tyre
on 11 of 13 frames (docs/stages/08-yolo-world-probe.md). The box is a bounding
box of a tracked mask, which is why the mask exists at all.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from excavator_cycles import masks as maskio
from excavator_cycles.onsets import derivative, motion_boundary, smooth
from excavator_cycles.geometry import otsu_threshold

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
SCRATCH = Path(__file__).parent
OUT = SCRATCH / "boxes.mp4"
VIDEO = REPO / "construction_excavator_cycle_duration_1.mp4"
FPS = 29.97396912419384
LABELS = {"digging": 125, "hauling": 322, "dumping": 559, "swinging": 691}
PHASES = list(LABELS)
COLOUR = {
    "digging": (90, 180, 250),
    "hauling": (120, 220, 120),
    "dumping": (250, 170, 70),
    "swinging": (220, 130, 240),
}
MACHINE_C = (150, 150, 150)
BUCKET_C = (40, 215, 255)

objects, shape = maskio.load_objects(REPO / "outputs/track/dual/masks.npz")
excavator, bucket = objects["excavator"], objects["bucket"]
MH, MW = shape


def box_of(mask):
    """The two lines that turn a tracked mask into a box."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---- the same detector the pipeline runs, so the labels cannot drift --------
sig = np.load(SCRATCH / "sig4.npz")
n = len(sig["h"])
t = np.arange(n) * 0.1
H, OV = smooth(sig["h"], t), smooth(sig["overlap"], t)
dH, dA = derivative(sig["h"], t), derivative(sig["angle"], t)
HS = smooth(sig["house_speed"], t)


def intervals(flag, times, min_s):
    out, start = [], None
    least = max(1, round(min_s / 0.1))
    for i in range(len(flag) + 1):
        on = i < len(flag) and flag[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= least:
                out.append((times[start], times[i - 1]))
            start = None
    return out


dig = max(intervals(H < otsu_threshold(H[np.isfinite(H)]), t, 1.0), key=lambda r: r[1] - r[0])
truck = max(intervals(OV > otsu_threshold(OV[np.isfinite(OV)]), t, 0.5), key=lambda r: r[1] - r[0])
mov = [m for m in intervals(HS > otsu_threshold(HS[np.isfinite(HS)]), t, 0.4) if m[0] > truck[0]]
T1 = motion_boundary(dH, t, (0.0, dig[1]), mode="arrives")
T2 = motion_boundary(dH, t, (dig[0], truck[0]), mode="departs")
T4 = motion_boundary(HS, t, (truck[0], mov[0][1]), mode="departs")
T3 = motion_boundary(dA, t, (truck[0], T4), mode="departs") if T4 and T4 > truck[0] + 0.5 else None
pred = {"digging": T1, "hauling": T2, "dumping": T3, "swinging": T4}
truth = {k: v / FPS for k, v in LABELS.items()}


def phase_at(now, table):
    marks = sorted(((k, v) for k, v in table.items() if v is not None), key=lambda kv: kv[1])
    active = None
    for name, start in marks:
        if now >= start:
            active = name
    return active


def label(canvas, text, origin, colour, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(canvas, (x, y - th - base - 3), (x + tw + 6, y), colour, cv2.FILLED)
    cv2.putText(canvas, text, (x + 3, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (20, 20, 20), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(str(VIDEO))
W = 1180
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * W / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
TOP, BOT = 92, 132
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, TOP + VH + BOT))

frame_no = -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    now = frame_no / FPS
    s = min(int(round(now / 0.1)), n - 1)

    view = cv2.resize(frame, (W, VH))
    sx, sy = W / MW, VH / MH

    for mask, colour, name, source in (
        (excavator.get(s), MACHINE_C, "machine", "SAM obj1  <- DINO 'excavator.'"),
        (bucket.get(s), BUCKET_C, "bucket", "SAM obj2  <- geodesic geometry"),
    ):
        box = box_of(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        p0 = (int(x0 * sx), int(y0 * sy))
        p1 = (int(x1 * sx), int(y1 * sy))
        thick = 3 if name == "bucket" else 2
        cv2.rectangle(view, p0, p1, colour, thick)
        label(view, f"{name}  {x1 - x0}x{y1 - y0}px", (p0[0], max(p0[1] - 3, 16)), colour, 0.48)
        if name == "bucket":  # the number T1/T2 actually read
            cv2.line(view, (p0[0], p1[1]), (p1[0], p1[1]), (60, 255, 255), 2)
            label(view, "box bottom -> h", (p1[0] + 6, p1[1] + 6), (60, 255, 255), 0.42)

    canvas = np.full((TOP + VH + BOT, W, 3), 14, np.uint8)
    canvas[TOP : TOP + VH] = view

    true_now, pred_now = phase_at(now, truth), phase_at(now, pred)
    cv2.putText(canvas, "TRUE", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (true_now or "-").upper(), (96, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLOUR.get(true_now, (190, 190, 190)), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRED", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (pred_now or "-").upper(), (96, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLOUR.get(pred_now, (190, 190, 190)), 2, cv2.LINE_AA)
    agree = true_now == pred_now
    cv2.putText(canvas, "MATCH" if agree else "MISMATCH", (W - 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (110, 220, 120) if agree else (90, 110, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={now:6.2f}s  f{frame_no}", (W - 430, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (140, 155, 175), 1, cv2.LINE_AA)

    y = TOP + VH + 26
    for line, colour in (
        ("WHERE THE BOXES COME FROM  -  no detector ever found the bucket", (150, 165, 185)),
        ("machine box = bbox of SAM 2 obj1, seeded once by Grounding DINO 'excavator.'", MACHINE_C),
        ("bucket  box = bbox of SAM 2 obj2, seeded once by GEODESIC GEOMETRY, then tracked", BUCKET_C),
        ("YOLO-World probe FAILED: landed on the dump truck's tyre on 11 of 13 frames", (90, 110, 240)),
    ):
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        y += 26

    writer.write(canvas)

cap.release()
writer.release()
print(f"wrote {OUT}")
print(f"T1 {T1}  T2 {T2}  T3 {T3}  T4 {T4}")
