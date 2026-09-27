"""TEMPORARY (fix/sam2-streaming) -- delete before merging.

Judge the cluster runs against the criteria fixed in the plan BEFORE any of them
ran, and print one PASS/FAIL line per criterion:

stage 3 -- same results (dev clip, 10 Hz, baseline = the previous tracker)
  masks vs previous tracker    median IoU >= 0.99 and min >= 0.95, both objects
  pruned vs every output kept  IoU == 1.0 on every sample, both objects
  two fresh streaming runs     identical masks, features and answer
  answer.json                  identical to the previous tracker's
  onset errors                 each |error| no more than 0.05 s worse

stage 4 -- flat memory
  streaming peak GPU memory at 5, 10 and 30 Hz within 20 % of each other,
  while the previous tracker's grows from 5 to 10 Hz

stage 6 -- the 83 s clip at 10 Hz completes

    compare.py outputs/sam2check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from excavator_cycles.config import Config
from excavator_cycles.features import load as load_features
from excavator_cycles.fsm import calibrate, locate, walk
from excavator_cycles.masks import load_objects

OUT = Path(sys.argv[1])
LABELS = Path("eval/labels.json")
failures: list[str] = []


def verdict(name: str, passed: bool, detail: str = "") -> None:
    if not passed:
        failures.append(name)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def objects(label: str) -> dict:
    return load_objects(OUT / label / "masks.npz")[0]


def per_object(x: dict, y: dict) -> dict:
    """IoU per sample, per object; a sample present in only one run scores 0."""
    out = {}
    for name in sorted(set(x) | set(y)):
        mx, my = x.get(name, {}), y.get(name, {})
        keys = sorted(set(mx) | set(my))
        scores = [iou(mx[k], my[k]) if k in mx and k in my else 0.0 for k in keys]
        out[name] = (np.array(scores), keys)
    return out


def onset_errors(label: str) -> dict:
    labels = json.loads(LABELS.read_text())
    fps = float(labels["video"]["fps"])
    keys = [
        "digging_begins",
        "hauling_begins",
        "dumping_begins",
        "swinging_begins",
        "cycle_ends",
    ]
    config = Config()
    table, _ = load_features(OUT / label)
    detections = walk(table, calibrate(table, config), config=config)
    onsets = locate(detections, table, config)
    times = table.time_seconds
    out = {}
    for key, detection, onset in zip(keys, detections, onsets, strict=False):
        truth = labels["boundaries"][key] / fps
        out[key] = {
            "fired": float(times[detection.fired_at]) - truth,
            "refined": None if onset.refined is None else onset.refined - truth,
        }
    return out


def ran(label: str) -> bool:
    peak = OUT / f"{label}.peak.json"
    if not peak.exists():
        verdict(f"{label} ran", False, "no record -- did it start?")
        return False
    record = json.loads(peak.read_text())
    good = record["exit"] in (0, 1) and record["error"] is None  # 1 = soft QA concern
    verdict(
        f"{label} ran",
        good,
        f"exit {record['exit']}  {record['seconds']} s  {record['error'] or ''}",
    )
    return good


def peak(label: str) -> float | None:
    path = OUT / f"{label}.peak.json"
    return json.loads(path.read_text())["peak_allocated_gib"] if path.exists() else None


print("== runs")
runs = [
    "dev_offline_10hz",
    "dev_stream_A_10hz",
    "dev_stream_B_10hz",
    "dev_keepall_10hz",
    "dev_offline_5hz",
    "dev_stream_5hz",
    "dev_stream_30hz",
    "long_stream_10hz",
]
ok = {label: ran(label) for label in runs}

base, a, b, keep = (
    "dev_offline_10hz",
    "dev_stream_A_10hz",
    "dev_stream_B_10hz",
    "dev_keepall_10hz",
)

print("\n== stage 3: same results (dev clip, 10 Hz)")
if ok[base] and ok[a]:
    for name, (v, keys) in per_object(objects(a), objects(base)).items():
        worst = keys[int(v.argmin())]
        verdict(
            f"{name} vs previous tracker",
            float(np.median(v)) >= 0.99 and float(v.min()) >= 0.95,
            f"n={len(v)} median {np.median(v):.4f} min {v.min():.4f} (sample {worst}) "
            f"below 0.99: {(v < 0.99).sum()}",
        )
    answer_a = json.loads((OUT / a / "answer.json").read_text())
    answer_base = json.loads((OUT / base / "answer.json").read_text())
    verdict(
        "answer.json identical to previous tracker",
        answer_a == answer_base,
        json.dumps(answer_a),
    )
    err_a, err_base = onset_errors(a), onset_errors(base)
    for key in err_base:
        for kind in ("fired", "refined"):
            x, y = err_a[key][kind], err_base[key][kind]
            if x is None and y is None:
                print(f"  [ -- ] {key} {kind}: neither located")
            elif x is None or y is None:
                verdict(
                    f"{key} {kind}", False, f"previous {y} streaming {x} (located in only one)"
                )
            else:
                verdict(
                    f"{key} {kind}",
                    abs(x) <= abs(y) + 0.05,
                    f"previous {y:+.3f} streaming {x:+.3f}",
                )
if ok[a] and ok[keep]:
    for name, (v, keys) in per_object(objects(a), objects(keep)).items():
        differing = [keys[i] for i in np.flatnonzero(v < 1.0)]
        verdict(
            f"{name} pruned == kept",
            not differing,
            f"n={len(v)} differing samples {differing[:8]}",
        )
if ok[a] and ok[b]:
    for name, (v, _keys) in per_object(objects(a), objects(b)).items():
        verdict(
            f"{name} run A == run B", bool((v == 1.0).all()), f"n={len(v)} min {v.min():.4f}"
        )
    for f in ("features.npz", "answer.json"):
        same = (OUT / a / f).read_bytes() == (OUT / b / f).read_bytes()
        verdict(f"{f} byte-identical A == B", same)

print("\n== stage 4: flat memory (peak GPU memory actually allocated, GiB)")
table = {
    "previous tracker": {5: peak("dev_offline_5hz"), 10: peak(base)},
    "streaming": {5: peak("dev_stream_5hz"), 10: peak(a), 30: peak("dev_stream_30hz")},
}
for name, row in table.items():
    print(
        f"  {name:18s} "
        + "   ".join(f"{hz:>2} Hz: {v if v is not None else '--'}" for hz, v in row.items())
    )
streams = [v for v in table["streaming"].values() if v is not None]
if len(streams) == 3:
    spread = (max(streams) - min(streams)) / min(streams)
    verdict("streaming flat across 5/10/30 Hz", spread < 0.20, f"spread {spread:.1%} (< 20 %)")
old = table["previous tracker"]
if old[5] is not None and old[10] is not None:
    verdict(
        "previous tracker grows 5 -> 10 Hz",
        old[10] > old[5] * 1.2,
        f"{old[5]} -> {old[10]} GiB",
    )

print("\n== stage 6: the 83 s clip at 10 Hz")
if ok["long_stream_10hz"]:
    answer = json.loads((OUT / "long_stream_10hz" / "answer.json").read_text())
    print(f"  peak {peak('long_stream_10hz')} GiB; answer {json.dumps(answer)}")

print(
    "\nRESULT:",
    "every criterion PASSES" if not failures else f"{len(failures)} FAIL: {failures}",
)
