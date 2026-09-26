"""Shared, CORRECT loading. masks.npz is keyed by SAMPLE ORDINAL, not frame index."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path("/Users/aanyashah/Desktop/github-repos/excavator-cycle-computer-vision")
sys.path.insert(0, str(REPO/"src"))
OUT = REPO/"outputs"/"track"/"real"
from excavator_cycles import masks as mask_io  # noqa: E402

def load():
    track = json.loads((OUT/"track.json").read_text())
    scene = json.loads((OUT/"scene.json").read_text())
    feats = dict(np.load(OUT/"features.npz", allow_pickle=True))
    masks, shape = mask_io.load(OUT/"masks.npz")
    records = track["frames"]
    assert len(records) == len(masks), (len(records), len(masks))
    # ordinal -> mask, ordinal -> frame index. THIS is the join that matters.
    frame_index = [r["frame_index"] for r in records]
    times = np.array([r["time_seconds"] for r in records])
    mask_list = [masks[o] for o in range(len(records))]
    want = set(frame_index)
    cap = cv2.VideoCapture(str(REPO/"construction_excavator_cycle_duration_1.mp4"))
    grays, colour = {}, {}
    i = 0
    while True:
        if not cap.grab(): break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); colour[i] = fr
        i += 1
    cap.release()
    return dict(track=track, scene=scene, feats=feats, shape=shape,
                frame_index=frame_index, times=times, mask_list=mask_list,
                grays=grays, colour=colour)
