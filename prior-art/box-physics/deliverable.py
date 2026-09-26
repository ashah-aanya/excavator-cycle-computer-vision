"""Render the progress video: masks + true/predicted phase + the signals that decide.

Saved as a file rather than a heredoc so the next run is a re-run, not a rewrite.

Two things this must not get wrong, both of which it got wrong before:
  * the playhead is placed through `ax.transData.transform`, never a guessed
    pixel formula -- a guessed one drifted ~0.7-1.0 s and read as a real error;
  * every trigger drawn here comes from the SAME code the pipeline uses
    (`onsets.motion_boundary`), so what you see is what is measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from excavator_cycles import masks as maskio
from excavator_cycles.geometry import otsu_threshold
from excavator_cycles.onsets import derivative, motion_boundary, noise_scale, smooth

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
SCRATCH = Path(__file__).parent
RUN = REPO / "outputs/track/dual"
VIDEO = REPO / "construction_excavator_cycle_duration_1.mp4"
OUT = SCRATCH / "deliverable_v3.mp4"

FPS_SRC = 29.97396912419384
LABELS = {"digging": 125, "hauling": 322, "dumping": 559, "swinging": 691}
PHASES = ["digging", "hauling", "dumping", "swinging"]
COLOUR = {
    "digging": (90, 180, 250),
    "hauling": (120, 220, 120),
    "dumping": (250, 170, 70),
    "swinging": (220, 130, 240),
}

# ---------------------------------------------------------------- signals ---
sig = np.load(SCRATCH / "sig4.npz")
n = len(sig["h"])
t = np.arange(n) * 0.1
H = smooth(sig["h"], t)
dH = derivative(sig["h"], t)
HS = smooth(sig["house_speed"], t)
OV = smooth(sig["overlap"], t)
dA = derivative(sig["angle"], t)


def intervals(flag, times, min_seconds):
    out, start = [], None
    least = max(1, round(min_seconds / float(np.median(np.diff(times)))))
    for i in range(len(flag) + 1):
        on = i < len(flag) and flag[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= least:
                out.append((times[start], times[i - 1]))
            start = None
    return out


# Pass 1: brackets, from Otsu on the signal's own distribution. No offsets.
dig = max(intervals(H < otsu_threshold(H[np.isfinite(H)]), t, 1.0), key=lambda r: r[1] - r[0])
truck = max(intervals(OV > otsu_threshold(OV[np.isfinite(OV)]), t, 0.5), key=lambda r: r[1] - r[0])
moving = [m for m in intervals(HS > otsu_threshold(HS[np.isfinite(HS)]), t, 0.4) if m[0] > truck[0]]

# Pass 2: the clock. Rest of a rate is zero; only the noise width is measured.
T1 = motion_boundary(dH, t, (0.0, dig[1]), mode="arrives")
T2 = motion_boundary(dH, t, (dig[0], truck[0]), mode="departs")
T4 = motion_boundary(HS, t, (truck[0], moving[0][1]), mode="departs")
T3 = motion_boundary(dA, t, (truck[0], T4), mode="departs") if T4 and T4 > truck[0] + 0.5 else None

pred = {"digging": T1, "hauling": T2, "dumping": T3, "swinging": T4}
truth = {k: v / FPS_SRC for k, v in LABELS.items()}

CUE = {
    "digging": ("dh/dt arrives 0", dH, "descent stops: bucket has reached the material"),
    "hauling": ("dh/dt departs 0", dH, "height leaves the dig plateau"),
    "dumping": ("d(angle)/dt departs 0", dA, "bucket starts rotating, over the truck"),
    "swinging": ("house speed departs 0", HS, "the house starts slewing back"),
}

print("bracket  dig %.1f-%.1f  truck %.1f-%.1f  swing %.1f-%.1f"
      % (dig[0], dig[1], truck[0], truck[1], moving[0][0], moving[0][1]))
for k in PHASES:
    p = pred[k]
    print(f"  {k:9s} {CUE[k][0]:24s} pred {('%.2f' % p) if p else 'None':>6s}  true {truth[k]:5.2f}"
          + (f"  err {p - truth[k]:+.2f}" if p else "  --"))

json.dump({"pred": pred, "true": truth}, open(SCRATCH / "onsets.json", "w"), indent=2)


def phase_at(time_s, table):
    """Which phase is active. Phases tile: each ends where the next begins."""
    marks = [(k, table[k]) for k in PHASES if table[k] is not None]
    marks.sort(key=lambda kv: kv[1])
    active = None
    for name, start in marks:
        if time_s >= start:
            active = name
    return active


# ------------------------------------------------------------------ panel ---
PANEL_W, PANEL_H = 620, 760
panels = [
    ("h  bucket height (L)", H, None),
    ("dh/dt  (L/s)   <- T1, T2", dH, 3 * noise_scale(dH)),
    ("house speed (px/s)   <- T4", HS, 3 * noise_scale(HS)),
    ("d(angle)/dt (rad/s)   <- T3", dA, 3 * noise_scale(dA)),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
fig.patch.set_facecolor("#0b0f16")
for ax, (label, values, band) in zip(axes, panels):
    ax.set_facecolor("#0b0f16")
    ax.plot(t, values, color="#e8edf5", lw=1.1)
    if band is not None:
        ax.axhline(0, color="#4a5568", lw=0.7)
        ax.axhspan(-band, band, color="#4a5568", alpha=0.35)
    for name in PHASES:
        ax.axvline(truth[name], color="#7f8c9b", lw=1.0, ls=":")
        if pred[name] is not None:
            ax.axvline(pred[name], color=np.array(COLOUR[name][::-1]) / 255.0, lw=1.4)
    ax.set_ylabel(label, color="#9fb0c4", fontsize=7)
    ax.tick_params(colors="#6b7a8d", labelsize=6)
    for s in ax.spines.values():
        s.set_color("#2a3341")
    ax.set_xlim(t[0], t[-1])
axes[-1].set_xlabel("seconds   (dotted = ground truth, solid = detected)", color="#9fb0c4", fontsize=7)
fig.tight_layout(pad=1.0)
fig.canvas.draw()

panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
# The playhead's x is taken from the AXES themselves. A guessed formula
# ((t/range)*(W-150)+92) drifted by ~0.7-1.0 s and was visible in the video.
x_of = [
    (
        float(axes[i].transData.transform((t[0], 0))[0]),
        float(axes[i].transData.transform((t[-1], 0))[0]),
        int(axes[i].get_window_extent().y0),
        int(axes[i].get_window_extent().y1),
    )
    for i in range(len(axes))
]
plt.close(fig)
PH, PW = panel.shape[:2]

# ------------------------------------------------------------------ video ---
# Through the LOADER, never np.load directly. masks.npz stores RUN LENGTHS, so
# reading it raw gives big positive integers and `m > 0` is true almost
# everywhere -- which tinted the entire frame, sky included.
objects, shape = maskio.load_objects(RUN / "masks.npz")
excavator = objects.get("excavator", {})
bucket = objects.get("bucket", {})
print(f"masks: {len(excavator)} excavator, {len(bucket)} bucket, shape {shape}")

cap = cv2.VideoCapture(str(VIDEO))
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
VID_W = 900
VID_H = int(src_h * VID_W / src_w)
TOP = 96
canvas_h = max(PH, VID_H + TOP)
writer = cv2.VideoWriter(
    str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS_SRC, (VID_W + PANEL_W, canvas_h)
)

fired: list[tuple[float, str]] = []
written = 0
frame_no = -1

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    now = frame_no / FPS_SRC
    sample = min(int(round(now / 0.1)), n - 1)

    view = cv2.resize(frame, (VID_W, VID_H))
    true_now = phase_at(now, truth)
    pred_now = phase_at(now, pred)

    # The excavator mask is tinted with the PHASE the detector currently
    # believes, so the machine itself reports the state -- a wrong call is
    # visible on the machine rather than only in a corner of the header.
    tone = COLOUR.get(pred_now, (170, 170, 170))
    m = excavator.get(sample)
    if m is not None and m.any():
        m = cv2.resize(m.astype(np.uint8), (VID_W, VID_H), interpolation=cv2.INTER_NEAREST)
        shade = view.copy()
        shade[m > 0] = tone
        view = cv2.addWeighted(shade, 0.38, view, 0.62, 0)
        edge, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, edge, -1, tone, 2)
    b = bucket.get(sample)
    if b is not None and b.any():
        b = cv2.resize(b.astype(np.uint8), (VID_W, VID_H), interpolation=cv2.INTER_NEAREST)
        overlay = view.copy()
        overlay[b > 0] = (60, 225, 255)
        view = cv2.addWeighted(overlay, 0.55, view, 0.45, 0)
        cont, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cont, -1, (40, 215, 255), 3)

    canvas = np.full((canvas_h, VID_W + PANEL_W, 3), 11, np.uint8)
    canvas[:, :, 1] = 15
    canvas[:, :, 2] = 22

    cv2.putText(canvas, "TRUE", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (true_now or "-").upper(), (110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                COLOUR.get(true_now, (200, 200, 200)), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRED", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (pred_now or "-").upper(), (110, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                COLOUR.get(pred_now, (200, 200, 200)), 2, cv2.LINE_AA)
    agree = true_now == pred_now
    cv2.putText(canvas, "MATCH" if agree else "MISMATCH", (VID_W - 190, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 220, 120) if agree else (90, 110, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={now:6.2f}s  f{frame_no}", (VID_W - 420, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 155, 175), 1, cv2.LINE_AA)

    canvas[TOP : TOP + VID_H, 0:VID_W] = view

    # trigger messages, latched as each fires
    for name in PHASES:
        p = pred[name]
        if p is not None and abs(now - p) < 0.5 / FPS_SRC:
            fired.append((p, f"{name.upper()}  {CUE[name][0]}  ->  {CUE[name][2]}"))
    y = TOP + VID_H + 26
    for when, text in fired[-4:]:
        cv2.putText(canvas, f"[{when:5.2f}s] {text}", (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (120, 200, 240), 1, cv2.LINE_AA)
        y += 24

    strip = panel.copy()
    for x0, x1, ytop, ybot in x_of:
        px = int(round(x0 + (now - t[0]) / (t[-1] - t[0]) * (x1 - x0)))
        cv2.line(strip, (px, PH - int(ybot)), (px, PH - int(ytop)), (90, 200, 255), 1)
    canvas[0:PH, VID_W : VID_W + PANEL_W] = strip

    writer.write(canvas)
    written += 1

cap.release()
writer.release()
print(f"wrote {OUT}  ({written} frames)")
