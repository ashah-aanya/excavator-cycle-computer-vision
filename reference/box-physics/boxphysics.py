"""Box-centre physics: trailing-averaged centres, the features, graphs and a marked video.

Smoothing is a TRAILING mean -- frame N is the mean of the w samples ending at N --
applied to the raw box centres before any physics. It is causal, so it lags by about
w/2; measured against the same window centred, that is +0.10 s at w=3 and +0.20 s at
w=5, and both score the same. Above w=9 both collapse, which is over-smoothing rather
than causality.

Nothing here uses optical flow.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from excavator_cycles import masks as maskio
from excavator_cycles.kinematics import body_core
from excavator_cycles.onsets import derivative as sg_deriv
from excavator_cycles.onsets import motion_boundary
from excavator_cycles.geometry import otsu_threshold

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
SD = Path(__file__).parent
W_SMOOTH = 5                      # 0.5 s trailing window
F = 29.97396912419384
TRUCK_BOX = (234.5, 134.0, 376.5, 211.0)
TRUTH = {"digging": 125 / F, "hauling": 322 / F, "dumping": 559 / F, "swinging": 691 / F}
COL = {"digging": "#5ab4f0", "hauling": "#78dc78", "dumping": "#f0aa46", "swinging": "#dc82f0"}

import json
scene = json.load(open(REPO / "outputs/track/real/scene.json"))
PIV = scene["centre"]; L = scene["scale"]
objects, shape = maskio.load_objects(REPO / "outputs/track/dual/masks.npz")
bucket, excavator = objects["bucket"], objects["excavator"]
core = body_core(list(excavator.values()))
n = 296
t = np.arange(n) * 0.1


def box_of(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


raw = {k: np.full(n, np.nan) for k in
       ("bx", "by", "ex", "ey", "bw", "bh", "ov", "bx0", "by0", "bx1", "by1",
        "ex0", "ey0", "ex1", "ey1")}
for i in range(n):
    b = bucket.get(i)
    if b is not None and b.any():
        x0, y0, x1, y1 = box_of(b)
        raw["bx"][i], raw["by"][i] = (x0 + x1) / 2, (y0 + y1) / 2   # mean of x, mean of y
        raw["bw"][i], raw["bh"][i] = x1 - x0 + 1, y1 - y0 + 1
        raw["bx0"][i], raw["by0"][i], raw["bx1"][i], raw["by1"][i] = x0, y0, x1, y1
        ix = max(0, min(x1, TRUCK_BOX[2]) - max(x0, TRUCK_BOX[0]))
        iy = max(0, min(y1, TRUCK_BOX[3]) - max(y0, TRUCK_BOX[1]))
        raw["ov"][i] = ix * iy / max(raw["bw"][i] * raw["bh"][i], 1)
    m = excavator.get(i)
    if m is not None and m.any():
        body = m & core
        if body.sum() > 30:
            x0, y0, x1, y1 = box_of(body)
            raw["ex"][i], raw["ey"][i] = (x0 + x1) / 2, (y0 + y1) / 2
            raw["ex0"][i], raw["ey0"][i], raw["ex1"][i], raw["ey1"][i] = x0, y0, x1, y1


def trailing(v, w=W_SMOOTH):
    """Frame N = mean of the w samples ENDING at N."""
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - w + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


s = {k: trailing(v) for k, v in raw.items()}

# --- the features, all in units of L (arm reach), from smoothed centres -----
X = s["bx"] / L
Y = s["by"] / L
EX = s["ex"] / L
EY = s["ey"] / L
RELX = X - EX                       # bucket MINUS body, x
RELY = EY - Y                       # bucket MINUS body, y (positive = bucket above)
H = (PIV[1] / L) - Y                # bucket height above the slew centre
dX = sg_deriv(X, t)
dY = sg_deriv(Y, t)
dH = sg_deriv(H, t)
ddH = sg_deriv(dH, t)
SPEEDX = np.abs(dX)
AR = s["bw"] / np.maximum(s["bh"], 1e-9)
dAR = sg_deriv(AR, t)
OV = s["ov"]
RAD = np.hypot(X - PIV[0] / L, Y - PIV[1] / L)


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

T1 = motion_boundary(dH, t, (0.0, dig[1]), mode="arrives")
T2 = motion_boundary(ddH, t, (lowest, truck[0]), mode="arrives")
T4 = motion_boundary(SPEEDX, t, (truck[0], mov[0][1]), mode="departs") if mov else None
# T3 = where the bucket's silhouette is MOST elongated. It is carried stretched
# out, peaks as it begins to tip, then collapses as it rotates over. Measured
# +0.55 s, against +1.75 s for "d(AR)/dt departs zero".
_tm = (t >= truck[0]) & (t <= truck[1])
T3 = float(t[np.nanargmax(np.where(_tm, AR, -np.inf))])
PRED = {"digging": T1, "hauling": T2, "dumping": T3, "swinging": T4}
print("onsets:", {k: (round(v, 2) if v else None) for k, v in PRED.items()})
for k in TRUTH:
    p = PRED[k]
    print(f"   {k:9s} {('%.2f' % p) if p else 'None':>6s} vs {TRUTH[k]:5.2f}"
          + (f"  {p - TRUTH[k]:+.2f}  {'PASS' if abs(p - TRUTH[k]) <= 0.6 else 'FAIL'}" if p else ""))

# ------------------------------------------------------------------ graphs --
PANELS = [
    ("bucket x  (L)", X, None),
    ("bucket y  (L)   down is +", Y, None),
    ("dx/dt  bucket  (L/s)", dX, 0.0),
    ("dy/dt  bucket  (L/s)", dY, 0.0),
    ("bucket - body  x  (L)", RELX, None),
    ("bucket - body  y  (L)   + = above", RELY, None),
    ("h = height above slew centre  (L)   -> T1, T2", H, None),
    ("|dx/dt|  -> T4", SPEEDX, 0.0),
    ("box aspect ratio w/h   PEAK -> T3", AR, None),
    ("bucket ^ truck box  -> T3 gate", OV, None),
    ("radius from slew centre  (L)", RAD, None),
]
fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 2.0 * len(PANELS)), dpi=110, sharex=True)
fig.patch.set_facecolor("#0b0f16")
for ax, (lab, v, zero) in zip(axes, PANELS):
    ax.set_facecolor("#0b0f16")
    ax.plot(t, v, color="#e8edf5", lw=1.2)
    if zero is not None:
        ax.axhline(zero, color="#54606f", lw=0.8)
    for name, tr in TRUTH.items():
        ax.axvline(tr, color="#8b98a8", lw=1.1, ls=":")
        if PRED[name] is not None:
            ax.axvline(PRED[name], color=COL[name], lw=1.6)
    ax.set_ylabel(lab, color="#a8b8cc", fontsize=8)
    ax.tick_params(colors="#70808f", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a3341")
axes[-1].set_xlabel("seconds    dotted = ground truth    solid = detected", color="#a8b8cc")
axes[0].set_title(f"Box-centre features, trailing mean over {W_SMOOTH} samples "
                  f"({W_SMOOTH * 0.1:.1f} s).  No optical flow.",
                  color="#e8edf5", fontsize=11)
fig.tight_layout()
fig.savefig(SD / "boxphysics.png", facecolor=fig.get_facecolor())
print("wrote boxphysics.png")
plt.close(fig)

# ------------------------------------------------------------------- video --
cap = cv2.VideoCapture(str(REPO / "construction_excavator_cycle_duration_1.mp4"))
VW = 1280
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * VW / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
TOP, BOT = 84, 96
writer = cv2.VideoWriter(str(SD / "boxphysics.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         F, (VW, TOP + VH + BOT))
MH, MW = shape
sx, sy = VW / MW, VH / MH


def phase_at(now, table):
    marks = sorted(((k, v) for k, v in table.items() if v is not None), key=lambda kv: kv[1])
    a = None
    for name, start in marks:
        if now >= start:
            a = name
    return a


fno = -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fno += 1
    now = fno / F
    i = min(int(round(now / 0.1)), n - 1)
    view = cv2.resize(frame, (VW, VH))

    P = lambda x, y: (int(x * sx), int(y * sy))  # noqa: E731
    cv2.rectangle(view, P(TRUCK_BOX[0], TRUCK_BOX[1]), P(TRUCK_BOX[2], TRUCK_BOX[3]),
                  (90, 110, 130), 1)
    cv2.putText(view, "truck bed box", P(TRUCK_BOX[0], TRUCK_BOX[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 110, 130), 1, cv2.LINE_AA)

    pv = P(PIV[0], PIV[1])
    cv2.drawMarker(view, pv, (120, 200, 120), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(view, "slew centre", (pv[0] + 8, pv[1] + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (120, 200, 120), 1, cv2.LINE_AA)

    if np.isfinite(s["ex"][i]):
        cv2.rectangle(view, P(s["ex0"][i], s["ey0"][i]), P(s["ex1"][i], s["ey1"][i]),
                      (170, 170, 170), 1)
        ec = P(s["ex"][i], s["ey"][i])
        cv2.drawMarker(view, ec, (220, 220, 220), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(view, "body centre", (ec[0] + 8, ec[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)

    if np.isfinite(s["bx"][i]):
        cv2.rectangle(view, P(s["bx0"][i], s["by0"][i]), P(s["bx1"][i], s["by1"][i]),
                      (40, 215, 255), 2)
        bc = P(s["bx"][i], s["by"][i])
        cv2.drawMarker(view, bc, (40, 215, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(view, "bucket centre", (bc[0] + 8, bc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (40, 215, 255), 1, cv2.LINE_AA)
        if np.isfinite(s["ex"][i]):  # the bucket-minus-body vector
            cv2.arrowedLine(view, P(s["ex"][i], s["ey"][i]), bc, (150, 255, 200), 1,
                            tipLength=0.04)
            cv2.putText(view, f"rel ({RELX[i]:+.2f}, {RELY[i]:+.2f}) L",
                        ((ec[0] + bc[0]) // 2 - 40, (ec[1] + bc[1]) // 2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 200), 1, cv2.LINE_AA)

    canvas = np.full((TOP + VH + BOT, VW, 3), 13, np.uint8)
    canvas[TOP:TOP + VH] = view
    tn, pn = phase_at(now, TRUTH), phase_at(now, PRED)
    bgr = lambda h: tuple(int(h[k:k + 2], 16) for k in (5, 3, 1))  # noqa: E731
    cv2.putText(canvas, "TRUE", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (tn or "-").upper(), (92, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                bgr(COL[tn]) if tn else (190, 190, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRED", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, (pn or "-").upper(), (92, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                bgr(COL[pn]) if pn else (190, 190, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MATCH" if tn == pn else "MISMATCH", (VW - 210, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (110, 220, 120) if tn == pn else (90, 110, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={now:6.2f}s f{fno}", (VW - 440, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (140, 155, 175), 1, cv2.LINE_AA)

    y = TOP + VH + 24
    for line, col in (
        (f"centres = mean(x), mean(y) of each box, trailing mean over {W_SMOOTH} samples "
         f"({W_SMOOTH * 0.1:.1f}s).  No optical flow.", (150, 165, 185)),
        (f"bucket ({X[i]:.2f}, {Y[i]:.2f}) L    dx/dt {dX[i]:+.3f}    dy/dt {dY[i]:+.3f} L/s"
         f"    h {H[i]:+.3f} L    overlap {OV[i]:.2f}", (40, 215, 255)),
        (f"bucket - body  x {RELX[i]:+.3f} L   y {RELY[i]:+.3f} L", (150, 255, 200)),
    ):
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        y += 24

    writer.write(canvas)

cap.release()
writer.release()
print("wrote boxphysics.mp4")
