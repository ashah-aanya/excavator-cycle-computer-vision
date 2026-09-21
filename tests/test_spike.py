"""Tests for the detector spike.

No model weights are downloaded here. A stub detector stands in for the real
one, which lets us test the part that is actually ours: the metrics, the gate,
and the artifacts. Whether Grounding DINO finds excavators is a question about
the world, answered by running the spike on real footage -- not something a unit
test can assert.

The stubs are written as *scenarios* -- a good detector, one that misses frames,
one that grabs the whole frame, one that jumps around -- because the gate's job
is to tell those apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.detect.base import Detection, contact_sheet, draw_detections, iou
from excavator_cycles.spike import evaluate_gate, format_report, run_spike


class StubDetector:
    """A detector with scripted behaviour, so the gate can be tested.

    Args:
        box: the box to return, as fractions of the frame.
        score: confidence to report.
        miss_every: return nothing on every Nth call (0 = never miss).
        jitter: random shift per call, as a fraction of frame width.
    """

    def __init__(
        self, name="stub", box=(0.3, 0.3, 0.6, 0.7), score=0.8, miss_every=0, jitter=0.0
    ):
        self.name = name
        self.box = box
        self.score = score
        self.miss_every = miss_every
        self.jitter = jitter
        self.calls = 0
        self._rng = np.random.default_rng(0)

    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        self.calls += 1
        if self.miss_every and self.calls % self.miss_every == 0:
            return Detection.empty(prompt)

        height, width = image.shape[:2]
        shift = self._rng.uniform(-self.jitter, self.jitter) if self.jitter else 0.0
        x1, y1, x2, y2 = self.box
        box = np.array(
            [(x1 + shift) * width, y1 * height, (x2 + shift) * width, y2 * height],
            dtype=np.float32,
        )
        return Detection(
            boxes=box[None, :],
            scores=np.array([self.score], dtype=np.float32),
            labels=[prompt],
            prompt=prompt,
        )


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A short synthetic video with visible structure, so overlays are checkable."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(150):  # 5 seconds
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (60 + i, 120), 20, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# --- geometry helpers -------------------------------------------------------


def test_iou_identical_boxes():
    box = np.array([10.0, 10.0, 20.0, 20.0])
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou(np.array([0.0, 0.0, 5.0, 5.0]), np.array([10.0, 10.0, 15.0, 15.0])) == 0.0


def test_iou_half_overlap():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 0.0, 15.0, 10.0])
    # intersection 50, union 150
    assert iou(a, b) == pytest.approx(1 / 3, abs=1e-6)


def test_detection_rejects_malformed_boxes():
    with pytest.raises(ValueError, match="boxes must be"):
        Detection(boxes=np.zeros((2, 3)), scores=np.zeros(2))
    with pytest.raises(ValueError, match="same length"):
        Detection(boxes=np.zeros((2, 4)), scores=np.zeros(3))


def test_empty_detection_is_falsy_but_valid():
    detection = Detection.empty("excavator.")
    assert not detection.found
    assert detection.best() is None
    assert len(detection) == 0


# --- drawing ----------------------------------------------------------------


def test_draw_detections_does_not_mutate_input():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = Detection(
        boxes=np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        labels=["excavator."],
    )
    drawn = draw_detections(image, detection, caption="t=1.0s")
    assert drawn.shape == image.shape
    assert image.sum() == 0, "the original frame must be left untouched"
    assert drawn.sum() > 0, "something should have been drawn"


def test_contact_sheet_tiles_all_frames():
    frames = [np.full((120, 160, 3), i, dtype=np.uint8) for i in range(7)]
    sheet = contact_sheet(frames, columns=3, tile_width=80)
    # 7 frames in rows of 3 -> 3 rows
    assert sheet.shape[1] == 80 * 3
    assert sheet.shape[0] > 0


def test_contact_sheet_rejects_empty():
    with pytest.raises(ValueError, match="no images"):
        contact_sheet([])


# --- the spike end to end ---------------------------------------------------


def test_spike_writes_all_artifacts(clip: Path, tmp_path: Path):
    out = tmp_path / "spike"
    report = run_spike(
        clip, {"stub": StubDetector()}, Config.load(), out, prompts=["excavator."], n_frames=6
    )

    assert (out / "metrics.json").exists()
    assert (out / "contact_sheet.jpg").exists()
    assert (out / "run.json").exists(), "provenance must be recorded"
    assert len(list((out / "frames").glob("*.jpg"))) == report.n_frames

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["gate"]["status"] == "pass"


def test_spike_samples_across_the_whole_video(clip: Path, tmp_path: Path):
    """Frames must be spread out, or the sample only sees one phase of the cycle."""
    detector = StubDetector()
    run_spike(
        clip, {"stub": detector}, Config.load(), tmp_path, prompts=["excavator."], n_frames=5
    )
    assert detector.calls == 5


def test_gate_fails_on_missed_frames(clip: Path, tmp_path: Path):
    """A detector that misses every third frame is not good enough to build on."""
    report = run_spike(
        clip,
        {"stub": StubDetector(miss_every=3)},
        Config.load(),
        tmp_path,
        prompts=["excavator."],
        n_frames=9,
    )
    assert report.gate["status"] == "fail"
    assert any("detection rate" in r for e in report.gate["checked"] for r in e["reasons"])


def test_gate_fails_when_box_swallows_the_frame(clip: Path, tmp_path: Path):
    """High confidence on a box covering everything is still a failure."""
    report = run_spike(
        clip,
        {"stub": StubDetector(box=(0.01, 0.01, 0.99, 0.99), score=0.95)},
        Config.load(),
        tmp_path,
        prompts=["excavator."],
        n_frames=5,
    )
    assert report.gate["status"] == "fail"
    assert any("whole scene" in r for e in report.gate["checked"] for r in e["reasons"])


def test_gate_fails_on_low_confidence(clip: Path, tmp_path: Path):
    report = run_spike(
        clip,
        {"stub": StubDetector(score=0.2)},
        Config.load(),
        tmp_path,
        prompts=["excavator."],
        n_frames=5,
    )
    assert report.gate["status"] == "fail"
    assert any("median score" in r for e in report.gate["checked"] for r in e["reasons"])


def test_gate_passes_if_any_prompt_works(clip: Path, tmp_path: Path):
    """A weak prompt is informative, not disqualifying -- the gate wants ONE winner."""
    config = Config.load()
    report = run_spike(
        clip,
        {"good": StubDetector(), "weak": StubDetector(score=0.1)},
        config,
        tmp_path,
        prompts=["excavator."],
        n_frames=5,
    )
    assert report.gate["status"] == "pass"
    assert report.gate["best"]["detector"] == "good"


def test_cross_model_agreement_reported(clip: Path, tmp_path: Path):
    """Two detectors pointing at the same object should show high IoU."""
    report = run_spike(
        clip,
        {"a": StubDetector(), "b": StubDetector(box=(0.31, 0.31, 0.61, 0.71))},
        Config.load(),
        tmp_path,
        prompts=["excavator."],
        n_frames=4,
    )
    agreement = report.cross_model_agreement
    assert agreement is not None
    assert agreement["median_iou"] > 0.8


def test_no_agreement_with_single_detector(clip: Path, tmp_path: Path):
    report = run_spike(
        clip, {"only": StubDetector()}, Config.load(), tmp_path, prompts=["x."], n_frames=3
    )
    assert report.cross_model_agreement is None


def test_report_formats_without_error(clip: Path, tmp_path: Path):
    report = run_spike(
        clip,
        {"stub": StubDetector()},
        Config.load(),
        tmp_path,
        prompts=["excavator."],
        n_frames=3,
    )
    text = format_report(report)
    assert "GATE: PASS" in text
    assert "contact_sheet.jpg" in text, "the reader must be told to look at the frames"


def test_gate_handles_no_results():
    """Degenerate input must not raise -- the pipeline never crashes on bad data."""
    gate = evaluate_gate([], Config.load())
    assert gate["status"] == "fail"
    assert gate["best"] is None
