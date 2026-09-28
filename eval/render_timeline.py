"""Render the whole-clip feature graphs with the pipeline's detections and the labels.

**Evaluation scaffolding, not pipeline code.** It runs the pipeline's state machine
(``calibrate``, ``walk``, ``locate``) on a clip's features -- the allowed direction
of import -- and draws, on every feature graph across the whole clip:

* a SHADED BAND in the phase's colour: the window in which the pipeline detected
  that phase (pass 1's window);
* a SOLID line: the pipeline's predicted onset (pass 2's refined time);
* a DOTTED line: the hand-labelled onset.

Writes a still image, and -- with ``--video`` -- a video of the footage above the
same graphs, with a white line marking the current moment.

Usage::

    uv run python eval/render_timeline.py                          # still image
    uv run python eval/render_timeline.py --video path/to/clip.mp4 # and the video
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
PHASES = ("digging", "hauling", "dumping", "swinging")
HEX = {"digging": "#E69F00", "hauling": "#56B4E9", "dumping": "#009E73", "swinging": "#CC79A7"}
NAMES = {"digging": "dig", "hauling": "haul", "dumping": "dump", "swinging": "return swing"}

PANELS = (
    ("bucket_x", "bucket x (L)"),
    ("bucket_y", "bucket y (L)\nimage, down +"),
    ("height", "height above\nslew centre (L)"),
    ("dh_dt", "dh/dt (L/s)"),
    ("d2h_dt2", "d2h/dt2 (L/s^2)"),
    ("dx_dt", "dx/dt (L/s)"),
    ("speed_x", "|dx/dt| (L/s)"),
    ("speed_2d", "2D speed (L/s)"),
    ("d2x_dt2", "d2x/dt2 (L/s^2)"),
    ("rel_cabin_x", "bucket - cabin x"),
    ("rel_cabin_y", "bucket - cabin y"),
    ("rel_truck_x", "bucket - truck x"),
    ("truck_overlap", "bucket ^ truck"),
    ("aspect_ratio", "box aspect w/h"),
    ("radius", "radius from\nslew centre"),
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detections(table):
    """Pass 1's windows and pass 2's times, exactly as the CLI runs them."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import calibrate, locate, walk

    config = Config.load(EVAL.parent / "configs" / "default.yaml")
    found = walk(table, calibrate(table, config), config=config)
    return found, locate(found, table, config)


def draw(t, feats, found, onsets, labelled, width_px=960, panel_px=44, dpi=100):
    """The graphs as one figure. Returns (figure, x pixel of t=0, x pixel of t=end)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    head_px = 46
    height_px = head_px + panel_px * len(PANELS) + 26
    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, sharex=True
    )
    fig.patch.set_facecolor("#0b0f16")
    fig.subplots_adjust(
        left=0.13, right=0.99, top=1 - head_px / height_px, bottom=26 / height_px, hspace=0.12
    )
    for ax, (key, label) in zip(axes, PANELS, strict=True):
        ax.set_facecolor("#0b0f16")
        for d in found:
            lo, hi = t[d.window.lo], t[min(d.window.hi, len(t)) - 1]
            # at least 0.3 s wide, so a window squeezed against the clip edge shows
            mid, half = (lo + hi) / 2, max((hi - lo) / 2, 0.15)
            ax.axvspan(mid - half, mid + half, color=HEX[d.phase], alpha=0.35, lw=0)
        ax.plot(t, feats[key], color="#e8edf5", lw=0.9)
        for o in onsets:
            if o.refined is not None:
                ax.axvline(o.refined, color=HEX[o.phase], lw=1.6)
        for phase, s in labelled:
            ax.axvline(s, color=HEX[phase], lw=1.4, ls=(0, (2, 2)))
        ax.set_ylabel(
            label, color="#a8b8cc", fontsize=6.5, rotation=0, ha="right", va="center"
        )
        ax.tick_params(colors="#70808f", labelsize=6, length=2)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a3341")
    # padded, so a line at exactly the first or last sample is not hidden on the frame
    axes[-1].set_xlim(t[0] - 0.8, t[-1] + 0.8)
    axes[-1].set_xlabel("seconds", color="#a8b8cc", fontsize=7, labelpad=1)
    handles = [Patch(color=HEX[p], alpha=0.6, label=NAMES[p]) for p in PHASES]
    handles += [
        Line2D([], [], color="#e8edf5", lw=1.6, label="prediction (solid)"),
        Line2D([], [], color="#e8edf5", lw=1.4, ls=(0, (2, 2)), label="your label (dotted)"),
        Patch(color="#8899aa", alpha=0.35, label="shaded = window it was detected in"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        fontsize=7,
        labelcolor="#e8edf5",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.canvas.draw()
    x0 = axes[0].transData.transform((t[0], 0))[0]
    x1 = axes[0].transData.transform((t[-1], 0))[0]
    return fig, x0, x1, height_px


def main(argv=None) -> int:
    cc = _load("check_cues")
    co = _load("check_onsets")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", choices=sorted(cc.CLIPS), default="long")
    ap.add_argument("--video", type=Path, help="the clip's video; omit for the still only")
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "timeline")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    feat_path, label_path = cc.CLIPS[args.clip]
    t, feats = cc.load_features(feat_path)
    labelled = cc.load_onsets(label_path)
    table = co.table_from(feat_path)
    found, onsets = detections(table)

    fig, x0, x1, graph_h = draw(t, feats, found, onsets, labelled)
    still = args.out.with_suffix(".png")
    fig.savefig(still, facecolor=fig.get_facecolor())
    print(f"wrote {still}")
    if args.video is None:
        return 0

    import cv2

    graph = cv2.imread(str(still))
    width = graph.shape[1]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"render_timeline: cannot open {args.video}", file=sys.stderr)
        return 2
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_h = round(fh * width / fw)
    size = (width, vid_h + graph.shape[0])
    out = args.out.with_suffix(".mp4")
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*codec), args.fps, size)
        if writer.isOpened():
            break
    boxes = np.load(feat_path)["bucket_box"] * (width / fw)
    bgr = {p: tuple(int(HEX[p][i : i + 2], 16) for i in (5, 3, 1)) for p in PHASES}
    next_t, written = 0.0, 0
    while cap.grab():
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if now + 1e-6 < next_t:
            continue
        next_t += 1.0 / args.fps
        _, frame = cap.retrieve()
        frame = cv2.resize(frame, (width, vid_h))
        i = int(np.argmin(np.abs(t - now)))
        labelled_now = next((p for p, s in reversed(labelled) if s <= now), None)
        predicted_now = next(
            (o.phase for o in reversed(onsets) if o.refined is not None and o.refined <= now),
            None,
        )
        x_a, y_a, x_b, y_b = boxes[i].astype(int)
        cv2.rectangle(
            frame, (x_a, y_a), (x_b, y_b), bgr.get(predicted_now, (200, 200, 200)), 2
        )
        for row, (who, phase) in enumerate(
            (("label", labelled_now), ("prediction", predicted_now))
        ):
            text = f"{who}: {NAMES.get(phase, '-')}"
            cv2.putText(
                frame,
                text,
                (12, 26 + 26 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (12, 26 + 26 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                bgr.get(phase, (230, 230, 230)),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"t = {now:5.1f} s",
            (width - 150, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panel = graph.copy()
        x = round(x0 + (now - t[0]) / (t[-1] - t[0]) * (x1 - x0))
        cv2.line(panel, (x, 40), (x, graph_h - 22), (255, 255, 255), 2)
        writer.write(np.vstack([frame, panel]))
        written += 1
    writer.release()
    print(f"wrote {out} ({written} frames at {args.fps:g} fps, {size[0]}x{size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
