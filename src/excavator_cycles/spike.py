"""Stage 1: does the detector actually find the excavator in *this* footage?

Everything downstream stands on this. The masks come from these boxes, the
geometry from the masks, the features from the geometry, the phase boundaries
from the features. If detection is unreliable, nothing built on top can be
trusted -- so this is checked first, cheaply, on a handful of frames, before any
of that exists.

Published zero-shot numbers cannot answer the question. Construction machinery
appears in no standard detection benchmark, and this is unusual footage: a
yellow machine against brown soil, partly self-occluding, in an environment
the model was never evaluated on. The only evidence that counts is this video.

The spike produces two kinds of evidence, and both matter:

* **Numbers** -- detection rate, score distribution, box stability -- which can
  be compared between prompts and between models, and checked against a gate.
* **Pictures** -- every sampled frame with its boxes drawn, plus a contact sheet.
  Metrics can look healthy while the box sits on a dump truck or a shadow. You
  have to look.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .detect import (
    Detector,
    box_area_fraction,
    box_centre,
    box_diagonal,
    contact_sheet,
    draw_detections,
    iou,
)
from .detect.base import Detection
from .logging_setup import get_logger
from .provenance import run_record
from .video import probe, read_frames_at, sample_times

log = get_logger(__name__)


@dataclass
class PromptResult:
    """How one (detector, prompt) pair performed across the sampled frames."""

    detector: str
    prompt: str

    frames: int = 0
    frames_with_detection: int = 0
    scores: list[float] = field(default_factory=list)
    area_fractions: list[float] = field(default_factory=list)
    centre_jumps: list[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.frames_with_detection / self.frames if self.frames else 0.0

    def summary(self) -> dict[str, Any]:
        """Condense to the numbers the gate is judged on."""

        def stats(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            array = np.asarray(values, dtype=np.float64)
            return {
                "min": float(array.min()),
                "median": float(np.median(array)),
                "mean": float(array.mean()),
                "max": float(array.max()),
            }

        return {
            "detector": self.detector,
            "prompt": self.prompt,
            "frames": self.frames,
            "frames_with_detection": self.frames_with_detection,
            "detection_rate": round(self.detection_rate, 4),
            "score": stats(self.scores),
            "area_fraction": stats(self.area_fractions),
            "centre_jump": stats(self.centre_jumps),
        }


@dataclass
class SpikeReport:
    """The spike's verdict, written to disk and printed."""

    video: str
    n_frames: int
    results: list[dict[str, Any]]
    cross_model_agreement: dict[str, Any] | None
    gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_spike(
    video_path: str | Path,
    detectors: dict[str, Detector],
    config: Config,
    output_dir: str | Path,
    prompts: list[str] | None = None,
    n_frames: int | None = None,
) -> SpikeReport:
    """Run every detector over every prompt on frames spread across the video.

    Args:
        video_path: the clip to test on.
        detectors: name -> detector. More than one enables the agreement check.
        config: thresholds and gate values.
        output_dir: where frames, contact sheets and metrics are written.
        prompts: phrases to test. Defaults to the configured excavator prompts.
        n_frames: how many frames to sample. Defaults to the configured value.

    Returns:
        The report, which is also written to ``metrics.json`` in ``output_dir``.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    prompts = list(prompts or config.detection.excavator_prompts)
    n_frames = n_frames or config.spike.n_frames

    info = probe(video_path)
    # Spread the sample across the whole clip so every phase of the cycle is
    # represented. Twenty consecutive frames would only show one phase, and
    # would say nothing about the moments where the arm is folded or occluded.
    times = sample_times(info, n_frames)
    samples = read_frames_at(video_path, times)
    log.info(
        "sampled %d frames from %s (%.1f s), testing %d prompt(s) x %d detector(s)",
        len(samples),
        video_path.name,
        info.duration_seconds,
        len(prompts),
        len(detectors),
    )

    with run_record("spike", output_dir, config.to_dict()) as record:
        record.add_input("video", video_path)

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        results: dict[tuple[str, str], PromptResult] = {
            (name, prompt): PromptResult(detector=name, prompt=prompt)
            for name in detectors
            for prompt in prompts
        }
        # Keep each frame's best box per detector, for the cross-model check and
        # for measuring how far the box moves between sampled frames.
        best_boxes: dict[str, list[np.ndarray | None]] = {name: [] for name in detectors}
        annotated: list[np.ndarray] = []

        for position, sample in enumerate(samples):
            per_frame: list[Detection] = []

            for name, detector in detectors.items():
                frame_best: np.ndarray | None = None
                frame_best_score = -1.0

                for prompt in prompts:
                    detection = detector.detect(sample.image, prompt)
                    result = results[(name, prompt)]
                    result.frames += 1

                    best = detection.best()
                    if best is not None:
                        box, score = best
                        result.frames_with_detection += 1
                        result.scores.append(score)
                        result.area_fractions.append(
                            box_area_fraction(box, sample.image.shape)
                        )
                        _record_centre_jump(result, box, position, best_boxes[name])
                        if score > frame_best_score:
                            frame_best, frame_best_score = box, score

                    per_frame.append(detection)

                best_boxes[name].append(frame_best)

            caption = f"t={sample.time_seconds:6.2f}s  frame {sample.frame_index}"
            canvas = draw_detections(sample.image, per_frame, caption=caption)
            annotated.append(canvas)
            cv2.imwrite(str(frames_dir / f"{position:03d}.jpg"), canvas)

        sheet_path = output_dir / "contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), contact_sheet(annotated))

        agreement = _cross_model_agreement(best_boxes)
        summaries = [result.summary() for result in results.values()]
        gate = evaluate_gate(summaries, config)

        report = SpikeReport(
            video=str(video_path),
            n_frames=len(samples),
            results=summaries,
            cross_model_agreement=agreement,
            gate=gate,
        )
        (output_dir / "metrics.json").write_text(json.dumps(report.to_dict(), indent=2))

        record.outputs.extend(["frames/", "contact_sheet.jpg", "metrics.json"])
        record.metrics = {"gate": gate["status"], "best": gate.get("best")}

    return report


def _record_centre_jump(
    result: PromptResult,
    box: np.ndarray,
    position: int,
    previous_boxes: list[np.ndarray | None],
) -> None:
    """How far the box moved since the previous sampled frame, relative to its size.

    Normalised by the box diagonal so the number means the same thing at any
    camera distance. Large values mean the detector is latching onto different
    things from frame to frame, which no amount of downstream smoothing fixes.
    """
    if position == 0 or position - 1 >= len(previous_boxes):
        return
    previous = previous_boxes[position - 1]
    if previous is None:
        return
    diagonal = box_diagonal(box)
    if diagonal <= 0:
        return
    jump = float(np.linalg.norm(box_centre(box) - box_centre(previous)) / diagonal)
    result.centre_jumps.append(jump)


def _cross_model_agreement(
    best_boxes: dict[str, list[np.ndarray | None]],
) -> dict[str, Any] | None:
    """Do two detectors point at the same object?

    Agreement is weak evidence of correctness -- two models can be wrong the
    same way -- but disagreement is strong evidence that at least one is wrong,
    and it says which frames to look at first.
    """
    names = [name for name, boxes in best_boxes.items() if any(b is not None for b in boxes)]
    if len(names) < 2:
        return None

    first, second = names[0], names[1]
    ious = [
        iou(a, b)
        for a, b in zip(best_boxes[first], best_boxes[second], strict=False)
        if a is not None and b is not None
    ]
    if not ious:
        return {"pair": [first, second], "frames_compared": 0, "median_iou": None}

    return {
        "pair": [first, second],
        "frames_compared": len(ious),
        "median_iou": round(float(np.median(ious)), 4),
        "min_iou": round(float(np.min(ious)), 4),
    }


def evaluate_gate(summaries: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Decide whether stage 1 passes, and say why.

    The gate is deliberately about the *best* prompt/detector combination: the
    question is whether ANY configuration detects the machine reliably, not
    whether all of them do. Weak prompts are informative, not disqualifying.
    """
    gate_config = config.spike
    checked: list[dict[str, Any]] = []

    for summary in summaries:
        reasons: list[str] = []
        if summary["detection_rate"] < gate_config.min_detection_rate:
            reasons.append(
                f"detection rate {summary['detection_rate']:.2f} "
                f"< {gate_config.min_detection_rate:.2f}"
            )
        score = summary["score"]
        if score is None or score["median"] < gate_config.min_median_score:
            median = "n/a" if score is None else f"{score['median']:.2f}"
            reasons.append(f"median score {median} < {gate_config.min_median_score:.2f}")
        area = summary["area_fraction"]
        if area is not None:
            if area["median"] > gate_config.max_box_area_fraction:
                reasons.append(
                    f"box covers {area['median']:.2f} of the frame "
                    f"(> {gate_config.max_box_area_fraction:.2f}); likely the whole scene"
                )
            if area["median"] < gate_config.min_box_area_fraction:
                reasons.append(
                    f"box covers only {area['median']:.4f} of the frame; likely background"
                )
        jump = summary["centre_jump"]
        if jump is not None and jump["median"] > gate_config.max_centre_jump:
            reasons.append(
                f"box centre moves {jump['median']:.2f} of its own size between frames"
            )

        checked.append({**summary, "passes": not reasons, "reasons": reasons})

    passing = [entry for entry in checked if entry["passes"]]
    if passing:
        best = max(
            passing, key=lambda entry: (entry["detection_rate"], entry["score"]["median"])
        )
        status = "pass"
    else:
        best = max(checked, key=lambda entry: entry["detection_rate"]) if checked else None
        status = "fail"

    return {
        "status": status,
        "best": None
        if best is None
        else {"detector": best["detector"], "prompt": best["prompt"]},
        "checked": checked,
    }


def format_report(report: SpikeReport) -> str:
    """Human-readable summary for the terminal."""
    lines = [
        f"Detector spike: {Path(report.video).name}  ({report.n_frames} frames sampled)",
        "",
        f"  {'detector':<16} {'prompt':<14} {'found':>7} {'score':>7} {'area':>7} {'jump':>7}",
        f"  {'-' * 16} {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}",
    ]
    for entry in report.gate["checked"]:
        score = "n/a" if entry["score"] is None else f"{entry['score']['median']:.2f}"
        area = (
            "n/a"
            if entry["area_fraction"] is None
            else f"{entry['area_fraction']['median']:.3f}"
        )
        jump = (
            "n/a" if entry["centre_jump"] is None else f"{entry['centre_jump']['median']:.2f}"
        )
        mark = "ok" if entry["passes"] else "XX"
        lines.append(
            f"  {entry['detector']:<16} {entry['prompt']:<14} "
            f"{entry['detection_rate']:>6.0%} {score:>7} {area:>7} {jump:>7}  {mark}"
        )

    if report.cross_model_agreement:
        agreement = report.cross_model_agreement
        lines += [
            "",
            f"  cross-model agreement ({' vs '.join(agreement['pair'])}): "
            f"median IoU {agreement['median_iou']}",
        ]

    lines += ["", f"  GATE: {report.gate['status'].upper()}"]
    if report.gate["status"] == "pass":
        best = report.gate["best"]
        lines.append(f"  best combination: {best['detector']} / {best['prompt']!r}")
    else:
        lines.append("  no prompt/detector combination met the gate. Reasons:")
        for entry in report.gate["checked"]:
            for reason in entry["reasons"]:
                lines.append(f"    - {entry['detector']}/{entry['prompt']}: {reason}")

    lines += [
        "",
        "  Look at contact_sheet.jpg before trusting these numbers: a healthy",
        "  detection rate with the box on the wrong object still fails.",
    ]
    return "\n".join(lines)
