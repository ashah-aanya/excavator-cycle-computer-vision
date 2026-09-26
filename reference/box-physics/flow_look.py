"""LOOK at the flow field across a progression of the video, before drawing conclusions.

Two artifacts:
  flow_progression.jpg  -- a grid of frames spanning the whole clip, flow drawn
  flow.mp4              -- every sampled frame, so the motion can be watched
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np

REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs" / "track" / "real"
SCRATCH = Path(sys.argv[1])
from excavator_cycles import masks as mask_io  # noqa: E402

track = json.loads((OUT / "track.json").read_text())
scene = json.loads((OUT / "scene.json").read_text())
centre = np.array(scene["centre"]); scale = float(scene["scale"])
masks, shape = mask_io.load(OUT / "masks.npz")
records = [f for f in track["frames"] if f["has_mask"]]
indices = [f["frame_index"] for f in records]
times = np.array([f["time_seconds"] for f in records])
wanted = set(indices)

cap = cv2.VideoCapture(str(REPO / "construction_excavator_cycle_duration_1.mp4"))
grays, colour = {}, {}
i = 0
while True:
    if not cap.grab(): break
    if i in wanted:
        ok, fr = cap.retrieve()
        if ok:
            grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); colour[i] = fr
    i += 1
cap.release()

dt = float(np.median(np.diff(times)))
UP = 3  # upscale for legibility
panels = []
for k in range(len(indices) - 1):
    a, b = indices[k], indices[k+1]
    if a not in grays or b not in grays: panels.append(None); continue
    m = masks.get(a)
    if m is None: panels.append(None); continue
    flow = cv2.calcOpticalFlowFarneback(grays[a], grays[b], None, 0.5, 4, 15, 3, 5, 1.2, 0)
    img = cv2.resize(colour[a], None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    # mask outline
    cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [c*UP for c in cnts], -1, (0,255,255), 1)
    cv2.circle(img, (int(centre[0]*UP), int(centre[1]*UP)), 4, (255,0,255), -1)
    # flow arrows on a grid inside the mask
    step = 6
    mags = []
    for y in range(0, shape[0], step):
        for x in range(0, shape[1], step):
            if not m[y, x]: continue
            fx, fy = flow[y, x]
            mag = float(np.hypot(fx, fy)); mags.append(mag)
            if mag < 0.15: 
                cv2.circle(img, (x*UP, y*UP), 1, (120,120,120), -1); continue
            g = 6.0  # arrow gain
            p0 = (x*UP, y*UP); p1 = (int((x+fx*g)*UP), int((y+fy*g)*UP))
            col = (0,255,0) if mag > 0.6 else (0,165,255)
            cv2.arrowedLine(img, p0, p1, col, 1, tipLength=0.35)
    mags = np.array(mags) if mags else np.zeros(1)
    hud = f"t={times[k]:5.2f}s  medflow={np.median(mags):.2f}px  p90={np.quantile(mags,0.9):.2f}px  moving={100*np.mean(mags>0.3):.0f}%"
    cv2.rectangle(img, (0,0), (img.shape[1], 16), (0,0,0), -1)
    cv2.putText(img, hud, (4,12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    panels.append(img)

# --- video over the whole progression
h, w = panels[0].shape[:2]
vw = cv2.VideoWriter(str(SCRATCH/"flow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w,h))
for p in panels:
    if p is not None: vw.write(p)
vw.release()

# --- contact sheet spanning the clip
good = [i for i, p in enumerate(panels) if p is not None]
picks = [good[int(j)] for j in np.linspace(0, len(good)-1, 12)]
tiles = [panels[p] for p in picks]
rows = [np.hstack(tiles[i:i+3]) for i in range(0, 12, 3)]
sheet = np.vstack(rows)
cv2.imwrite(str(SCRATCH/"flow_progression.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", SCRATCH/"flow.mp4", SCRATCH/"flow_progression.jpg", sheet.shape)
