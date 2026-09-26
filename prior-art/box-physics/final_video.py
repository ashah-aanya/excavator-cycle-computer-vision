"""Deliverable: tracked boxes + points on the left, the feature graphs on the right,
trigger points labelled on the graphs and logged as they fire.

Settings are the ones the 2-D sweep found: trailing mean 0.3 s on the raw box
centres, Savitzky-Golay 0.3 s for every derivative. Heavier smoothing lags each
transition by an amount that depends on that signal's own shape, which
manufactures differential bias and costs a graded field.

No optical flow anywhere. Every number comes from a bounding box.
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
from excavator_cycles.kinematics import body_core
from excavator_cycles.onsets import derivative, motion_boundary

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
SD = Path(__file__).parent
W_TRAIL, SG = 3, 0.3
F = 29.97396912419384
TRUCK_BOX = (234.5, 134.0, 376.5, 211.0)
TRUTH = {"digging": 125 / F, "hauling": 322 / F, "dumping": 559 / F, "swinging": 691 / F}
PHASES = list(TRUTH)
HEX = {"digging": "#5ab4f0", "hauling": "#78dc78", "dumping": "#f0aa46", "swinging": "#dc82f0"}
BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in HEX.items()}

scene = json.load(open(REPO / "outputs/track/real/scene.json"))
PIV, L = scene["centre"], scene["scale"]
objects, shape = maskio.load_objects(REPO / "outputs/track/dual/masks.npz")
bucket, excavator = objects["bucket"], objects["excavator"]
core = body_core(list(excavator.values()))
n, MH, MW = 296, *shape
t = np.arange(n) * 0.1

KEYS = ("bx", "by", "bw", "bh", "ov", "bx0", "by0", "bx1", "by1",
        "ex", "ey", "ex0", "ey0", "ex1", "ey1")
raw = {k: np.full(n, np.nan) for k in KEYS}
for i in range(n):
    b = bucket.get(i)
    if b is not None and b.any():
        ys, xs = np.nonzero(b)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        raw["bx"][i], raw["by"][i] = (x0 + x1) / 2, (y0 + y1) / 2
        raw["bw"][i], raw["bh"][i] = x1 - x0 + 1, y1 - y0 + 1
        raw["bx0"][i], raw["by0"][i], raw["bx1"][i], raw["by1"][i] = x0, y0, x1, y1
        ix = max(0, min(x1, TRUCK_BOX[2]) - max(x0, TRUCK_BOX[0]))
        iy = max(0, min(y1, TRUCK_BOX[3]) - max(y0, TRUCK_BOX[1]))
        raw["ov"][i] = ix * iy / max(raw["bw"][i] * raw["bh"][i], 1)
    m = excavator.get(i)
    if m is not None and m.any():
        body = m & core
        if body.sum() > 30:
            ys, xs = np.nonzero(body)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            raw["ex"][i], raw["ey"][i] = (x0 + x1) / 2, (y0 + y1) / 2
            raw["ex0"][i], raw["ey0"][i], raw["ex1"][i], raw["ey1"][i] = x0, y0, x1, y1


def trail(v, w=W_TRAIL):
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - w + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


s = {k: trail(v) for k, v in raw.items()}
D = lambda v: derivative(v, t, window_seconds=SG)   # noqa: E731

X, Y = s["bx"] / L, s["by"] / L
EX, EY = s["ex"] / L, s["ey"] / L
RELX, RELY = X - EX, EY - Y
H = (PIV[1] / L) - Y
dX, dY = D(X), D(Y)
SPEEDX = np.abs(dX)
AR = s["bw"] / np.maximum(s["bh"], 1e-9)
OV = s["ov"]
dH = D(H)
ddH = D(dH)


def iv(f, times, ms):
    out, st = [], None
    least = max(1, round(ms / 0.1))
    for i in range(len(f) + 1):
        on = i < len(f) and f[i]
        if on and st is None:
            st = i
        elif not on and st is not None:
            if i - st >= least:
                out.append((times[st], times[i - 1]))
            st = None
    return out


eps = iv(H < otsu_threshold(H[np.isfinite(H)]), t, 1.0)
dig = max(eps, key=lambda r: r[1] - r[0])
truck = max(iv(OV > otsu_threshold(OV[np.isfinite(OV)]), t, 0.5), key=lambda r: r[1] - r[0])
mov = [m for m in iv(SPEEDX > otsu_threshold(SPEEDX[np.isfinite(SPEEDX)]), t, 0.4)
       if m[0] > truck[0]]
mm = (t >= dig[0]) & (t <= dig[1])
lowest = float(t[np.nanargmin(np.where(mm, H, np.inf))])
tm = (t >= truck[0]) & (t <= truck[1])

T1 = motion_boundary(dH, t, (0.0, dig[1]), mode="arrives")
T2 = motion_boundary(ddH, t, (lowest, truck[0]), mode="arrives")
T3 = float(t[np.nanargmax(np.where(tm, AR, -np.inf))])
T4 = motion_boundary(SPEEDX, t, (truck[0], mov[0][1]), mode="departs")
PRED = {"digging": T1, "hauling": T2, "dumping": T3, "swinging": T4}

# which panel each trigger is read from, and the message it fires
TRIG = {
    "digging":  (6, "dh/dt ARRIVES 0", "bucket stops descending: it has reached the material"),
    "hauling":  (6, "d2h/dt2 ARRIVES 0", "lift stops accelerating: bucket is free of the material"),
    "dumping":  (7, "aspect ratio PEAKS", "silhouette most elongated: the bucket begins to tip"),
    "swinging": (5, "|dx/dt| DEPARTS 0", "bucket starts sweeping sideways: the house slews back"),
}
for k in PHASES:
    print(f"{k:9s} {PRED[k]:6.2f} vs {TRUTH[k]:5.2f}  {PRED[k] - TRUTH[k]:+.2f}"
          f"  {'PASS' if abs(PRED[k] - TRUTH[k]) <= 0.6 else 'FAIL'}   {TRIG[k][1]}")

PANELS = [
    ("bucket x", X),
    ("bucket y (down+)", Y),
    ("dx/dt", dX),
    ("dy/dt", dY),
    ("bucket-body x/y", None),
    ("|dx/dt| -> T4", SPEEDX),
    ("h -> T1,T2", H),
    ("aspect w/h -> T3", AR),
    ("truck overlap", OV),
]
PW, PH = 660, 980
fig, axes = plt.subplots(len(PANELS), 1, figsize=(PW / 100, PH / 100), dpi=100, sharex=True)
fig.patch.set_facecolor("#0b0f16")
for ax, (lab, v) in zip(axes, PANELS):
    ax.set_facecolor("#0b0f16")
    if v is None:
        ax.plot(t, RELX, color="#96ffc8", lw=1.1, label="x")
        ax.plot(t, RELY, color="#ffc896", lw=1.1, label="y")
        ax.legend(fontsize=5, loc="upper right", facecolor="#0b0f16",
                  edgecolor="#2a3341", labelcolor="#a8b8cc")
    else:
        ax.plot(t, v, color="#e8edf5", lw=1.1)
    for name, tr in TRUTH.items():
        ax.axvline(tr, color="#8b98a8", lw=1.0, ls=":")
        ax.axvline(PRED[name], color=HEX[name], lw=1.5)
    ax.set_ylabel(lab, color="#a8b8cc", fontsize=7.5)
    ax.tick_params(colors="#70808f", labelsize=5.5)
    for sp in ax.spines.values():
        sp.set_color("#2a3341")
axes[-1].set_xlabel("seconds   dotted = truth   solid = detected", color="#a8b8cc", fontsize=7)

# a labelled marker on the panel each trigger is actually READ from
SERIES = {5: SPEEDX, 6: H, 7: AR}
for name, (panel, rule, _) in TRIG.items():
    ax = axes[panel]
    yv = float(SERIES[panel][min(int(round(PRED[name] / 0.1)), n - 1)])
    ax.plot([PRED[name]], [yv], "o", color=HEX[name], ms=6, mec="w", mew=0.8, zorder=5)
    ax.annotate(f" {name[:4].upper()} {rule}", (PRED[name], yv), color=HEX[name],
                fontsize=5.5, xytext=(4, 4), textcoords="offset points", zorder=5)
fig.tight_layout(pad=0.9)
fig.canvas.draw()
panel_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
AX = [(float(a.transData.transform((t[0], 0))[0]),
       float(a.transData.transform((t[-1], 0))[0]),
       int(a.get_window_extent().y0), int(a.get_window_extent().y1)) for a in axes]
plt.close(fig)
PHh, PWw = panel_img.shape[:2]

cap = cv2.VideoCapture(str(SD / "boxphysics.mp4"))
SRC_TOP, SRC_BOT = 84, 96
VW = 1000
SRC_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - SRC_TOP - SRC_BOT
SRC_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(SRC_H * VW / SRC_W)
TOP, BOT = 88, 124
CH = max(PHh, TOP + VH + BOT)
writer = cv2.VideoWriter(str(SD / "scrub.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                         F, (VW + PWw, CH))
sx, sy = VW / MW, VH / MH


def phase_at(now, table):
    a = None
    for name, start in sorted(table.items(), key=lambda kv: kv[1]):
        if now >= start:
            a = name
    return a


fired, fno = [], -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fno += 1
    now = fno / F
    i = min(int(round(now / 0.1)), n - 1)
    view = cv2.resize(frame[SRC_TOP:SRC_TOP + SRC_H], (VW, VH))
    P = lambda x, y: (int(x * sx), int(y * sy))   # noqa: E731

    canvas = np.full((CH, VW + PWw, 3), 13, np.uint8)
    canvas[TOP:TOP + VH, :VW] = view
    tn, pn = phase_at(now, TRUTH), phase_at(now, PRED)
    cv2.putText(canvas, "TRUE", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (tn or "-").upper(), (92, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                BGR.get(tn, (190, 190, 190)), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRED", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (pn or "-").upper(), (92, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                BGR.get(pn, (190, 190, 190)), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MATCH" if tn == pn else "MISMATCH", (VW - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (110, 220, 120) if tn == pn else (90, 110, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={now:6.2f}s f{fno}", (VW - 430, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (140, 155, 175), 1, cv2.LINE_AA)

    for name in PHASES:
        if abs(now - PRED[name]) < 0.5 / F:
            fired.append((PRED[name], name))
    y = TOP + VH + 24
    for when, name in fired[-4:]:
        cv2.putText(canvas, f"[{when:5.2f}s] {name.upper():9s} {TRIG[name][1]:22s} -> {TRIG[name][2]}",
                    (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR[name], 1, cv2.LINE_AA)
        y += 22
    cv2.putText(canvas, f"boxes only, no optical flow | trailing {W_TRAIL * 0.1:.1f}s + savgol {SG:.1f}s"
                f" | bucket ({X[i]:.2f},{Y[i]:.2f})L  dx/dt {dX[i]:+.3f}  dy/dt {dY[i]:+.3f}",
                (16, CH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 145, 165), 1, cv2.LINE_AA)

    strip = panel_img.copy()
    for x0, x1, y0a, y1a in AX:
        px = int(round(x0 + (now - t[0]) / (t[-1] - t[0]) * (x1 - x0)))
        cv2.line(strip, (px, PHh - int(y1a)), (px, PHh - int(y0a)), (255, 255, 255), 2)
    canvas[0:PHh, VW:VW + PWw] = strip
    writer.write(canvas)

cap.release()
writer.release()
fig2, _ = plt.subplots()
plt.close(fig2)
print(f"wrote {SD / 'scrub.mp4'}")
