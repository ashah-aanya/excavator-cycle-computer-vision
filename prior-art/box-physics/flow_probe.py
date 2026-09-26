"""Probe: does dense optical flow inside the mask measure the machine's motion?

Pass criterion, fixed BEFORE running (see the conversation):
  * flow-derived angular rate vs chain-derived slew rate: correlation > 0.8 pass,
    < 0.5 abandon.
  * clean sign flips in the angular rate at the swings visible by eye.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs" / "track" / "real"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

from excavator_cycles import masks as mask_io  # noqa: E402

track = json.loads((OUT / "track.json").read_text())
scene = json.loads((OUT / "scene.json").read_text())
feats = np.load(OUT / "features.npz", allow_pickle=True)

centre = np.array(scene["centre"], dtype=float)
scale = float(scene["scale"])
masks, shape = mask_io.load(OUT / "masks.npz")

records = [f for f in track["frames"] if f["has_mask"]]
indices = [f["frame_index"] for f in records]
times = np.array([f["time_seconds"] for f in records])
wanted = set(indices)

# Sequential decode: grab everything, retrieve only the sampled frames.
cap = cv2.VideoCapture(str(REPO / track["video"].split("/")[-1]))
if not cap.isOpened():
    cap = cv2.VideoCapture(track["video"])
grays: dict[int, np.ndarray] = {}
colour: dict[int, np.ndarray] = {}
i = 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if i in wanted:
        ok, frame = cap.retrieve()
        if ok:
            grays[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            colour[i] = frame
    i += 1
cap.release()
print(f"decoded {len(grays)} of {len(wanted)} sampled frames")

dt = float(np.median(np.diff(times)))
ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
rx = xs - centre[0]
ry = ys - centre[1]
r2 = rx**2 + ry**2
r = np.sqrt(r2)

n = len(indices)
omega = np.full(n, np.nan)          # angular rate about the pivot, rad/s
v_rad = np.full(n, np.nan)          # reaching out (+) / pulling in (-), L/s
v_up = np.full(n, np.nan)           # distal vertical rate, L/s, up positive
speed = np.full(n, np.nan)          # median flow magnitude, L/s
n_px = np.zeros(n, dtype=int)
flows: dict[int, np.ndarray] = {}

for k in range(n - 1):
    a, b = indices[k], indices[k + 1]
    if a not in grays or b not in grays:
        continue
    m = masks.get(a)
    if m is None or m.sum() < 200:
        continue
    # Erode: flow on the silhouette edge mixes machine and background.
    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if inner.sum() < 150:
        inner = m
    flow = cv2.calcOpticalFlowFarneback(
        grays[a], grays[b], None,
        pyr_scale=0.5, levels=4, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    flows[a] = flow
    fx, fy = flow[..., 0], flow[..., 1]
    sel = inner & (r > 1.0)
    n_px[k] = int(sel.sum())
    if n_px[k] < 100:
        continue
    # Bearing convention in features.py is atan2(-(y-cy), x-cx): y flipped.
    w = (ry[sel] * fx[sel] - rx[sel] * fy[sel]) / r2[sel] / dt
    omega[k] = float(np.median(w))
    v_rad[k] = float(np.median((rx[sel] * fx[sel] + ry[sel] * fy[sel]) / r[sel])) / scale / dt
    speed[k] = float(np.median(np.hypot(fx[sel], fy[sel]))) / scale / dt
    # Distal fifth: the far end of the arm, where the bucket is.
    cut = np.quantile(r[sel], 0.8)
    distal = sel & (r >= cut)
    if distal.sum() > 30:
        v_up[k] = float(np.median(-fy[distal])) / scale / dt

np.savez(SCRATCH / "flow_probe.npz", time=times, omega=omega, v_rad=v_rad,
         v_up=v_up, speed=speed, n_px=n_px, indices=np.array(indices))

# --- the comparison the probe exists to make -------------------------------
slew = feats["slew_rate"]
valid = feats["valid"].astype(bool)
ok = np.isfinite(omega) & np.isfinite(slew) & valid
print(f"\ncomparable samples: {ok.sum()} of {n}")
if ok.sum() > 10:
    c = float(np.corrcoef(omega[ok], slew[ok])[0, 1])
    print(f"correlation(flow omega, chain slew) = {c:.3f}")
    print(f"  flow  omega: median |w| = {np.median(np.abs(omega[ok])):.4f} rad/s, "
          f"p95 = {np.quantile(np.abs(omega[ok]), 0.95):.4f}")
    print(f"  chain slew : median |w| = {np.median(np.abs(slew[ok])):.4f} rad/s, "
          f"p95 = {np.quantile(np.abs(slew[ok]), 0.95):.4f}")
    # Agreement on DIRECTION only, when either is clearly moving.
    moving = ok & (np.abs(slew) > np.quantile(np.abs(slew[ok]), 0.5))
    agree = float((np.sign(omega[moving]) == np.sign(slew[moving])).mean())
    print(f"  sign agreement while slewing: {agree:.1%} over {moving.sum()} samples")

# How noisy is each, frame to frame? This is the real question.
def jitter(x):
    d = np.diff(x[np.isfinite(x)])
    return float(np.median(np.abs(d))), float(np.quantile(np.abs(d), 0.95))

print(f"\nframe-to-frame change (median, p95):")
print(f"  flow  omega: {jitter(omega)}")
print(f"  chain slew : {jitter(slew[valid])}")
