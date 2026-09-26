"""Just the mask, filled, on the frame. No flow, no chain. Across the whole clip."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
sys.path.insert(0, str(REPO/"src"))
OUT = REPO/"outputs"/"track"/"real"
SCRATCH = Path(sys.argv[1])
from excavator_cycles import masks as mask_io

track = json.loads((OUT/"track.json").read_text())
masks, shape = mask_io.load(OUT/"masks.npz")
records = [f for f in track["frames"] if f["has_mask"]]
idx = [f["frame_index"] for f in records]
tms = [f["time_seconds"] for f in records]
want = set(idx)
cap = cv2.VideoCapture(str(REPO/"construction_excavator_cycle_duration_1.mp4"))
frames = {}
i = 0
while True:
    if not cap.grab(): break
    if i in want:
        ok, fr = cap.retrieve()
        if ok: frames[i] = fr
    i += 1
cap.release()

picks = np.linspace(0, len(idx)-1, 16).astype(int)
tiles = []
for p in picks:
    a = idx[p]
    if a not in frames: continue
    img = cv2.resize(frames[a], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    m = cv2.resize(masks[a].astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = img.copy(); overlay[m] = (0, 0, 255)
    img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.rectangle(img, (0,0), (img.shape[1],16), (0,0,0), -1)
    cv2.putText(img, f"t={tms[p]:.2f}s frame={a} area={masks[a].mean()*100:.1f}%",
                (4,12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    tiles.append(img)
rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles)-3, 4)]
sheet = np.vstack(rows)
sheet = cv2.resize(sheet, (1800, int(1800*sheet.shape[0]/sheet.shape[1])))
cv2.imwrite(str(SCRATCH/"mask_check.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", sheet.shape)
