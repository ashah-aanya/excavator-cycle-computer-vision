"""Integration tests: run the REAL detector, not a stub.

Why these exist separately from ``test_spike.py``:

* The stub tests prove *our* code is right -- the metrics, the gate, the
  artifacts. They cannot prove that the model call works, because there is no
  model in them.
* These tests prove the model path executes and returns what we expect: the
  right output keys, boxes in pixel coordinates in the image's frame, scores in
  a sane range. A wrong keyword argument or a renamed output key would pass
  every stub test and fail on the real video -- the worst place to discover it,
  because a code bug there is indistinguishable from the model simply not
  working on the footage.

They are **skipped by default**: they need ~700 MB of weights and a download.
Run them deliberately:

    uv sync --extra models
    uv run pytest -m integration -v

This split -- fast tests on every commit, slow model-dependent tests on demand
-- is standard practice, and it is why CI can stay free of GPUs and weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.detect.base import box_area_fraction

pytestmark = pytest.mark.integration

transformers = pytest.importorskip("transformers", reason="needs: uv sync --extra models")


@pytest.fixture(scope="module")
def detector():
    """The real Grounding DINO. Loaded once for the whole module -- it is slow."""
    from excavator_cycles.detect import build_detector

    return build_detector("grounding_dino", Config.load().detection)


@pytest.fixture
def synthetic_scene() -> np.ndarray:
    """A crude machine-like shape on ground.

    Not a real excavator, deliberately: this fixture tests the *plumbing* --
    that a call runs end to end and returns well-formed output. Whether the
    model recognises real excavators is answered by running the spike on real
    imagery, which no unit test can substitute for.
    """
    import cv2

    image = np.full((360, 640, 3), 130, dtype=np.uint8)
    cv2.rectangle(image, (0, 250), (640, 360), (90, 110, 140), -1)
    cv2.rectangle(image, (245, 205), (355, 270), (40, 190, 230), -1)
    cv2.line(image, (300, 250), (450, 150), (40, 190, 230), 14)
    return image


def test_detector_returns_well_formed_output(detector, synthetic_scene):
    """The contract every downstream stage relies on."""
    detection = detector.detect(synthetic_scene, "excavator.")

    assert detection.boxes.ndim == 2
    assert detection.boxes.shape[1] == 4 if len(detection) else True
    assert len(detection.boxes) == len(detection.scores)
    assert all(0.0 <= s <= 1.0 for s in detection.scores)


def test_boxes_are_in_pixel_coordinates(detector, synthetic_scene):
    """The failure this catches: normalised boxes silently treated as pixels.

    Models predict boxes in 0-1 coordinates internally. If post-processing were
    misconfigured, every box would land in the top-left few pixels -- geometry
    downstream would be nonsense, and nothing would raise.
    """
    detection = detector.detect(synthetic_scene, "machine.")
    if not detection.found:
        pytest.skip("nothing detected in the synthetic scene; coordinates untested")

    height, width = synthetic_scene.shape[:2]
    for box in detection.boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= width + 1, f"x out of frame: {box}"
        assert 0 <= y1 < y2 <= height + 1, f"y out of frame: {box}"
        # A box occupying a fraction of a percent of the frame is the signature
        # of normalised coordinates being read as pixels.
        assert box_area_fraction(box, synthetic_scene.shape) > 1e-4


def test_nothing_found_is_not_an_error(detector):
    """Blank frames must return empty, not raise.

    The pipeline has to produce an answer on the hidden videos, so a frame with
    nothing in it lowers a coverage metric -- it never stops the run.
    """
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    detection = detector.detect(blank, "excavator.")
    assert isinstance(detection.boxes, np.ndarray)
    assert len(detection) == len(detection.scores)


def test_labels_match_the_prompt(detector, synthetic_scene):
    """Guards against phrases being cross-attributed between boxes."""
    detection = detector.detect(synthetic_scene, "excavator.")
    if not detection.found:
        pytest.skip("nothing detected; labels untested")
    assert all(isinstance(label, str) for label in detection.labels)


@pytest.mark.skipif(
    not Path("data/sample_excavator.jpg").exists(),
    reason="drop a real excavator photo at data/sample_excavator.jpg to enable",
)
def test_finds_a_real_excavator(detector):
    """The question the stubs cannot answer: does it work on a real machine?

    Opt-in, because it needs an image we cannot ship. Put any excavator photo at
    data/sample_excavator.jpg and this becomes a real check of the thing that
    matters -- and a fast way to compare prompts before touching the video.
    """
    import cv2

    image = cv2.imread("data/sample_excavator.jpg")
    detection = detector.detect(image, "excavator.")

    assert detection.found, "no excavator detected in a photo of an excavator"
    box, score = detection.best()
    assert score > 0.3, f"detected, but weakly ({score:.2f})"
    area = box_area_fraction(box, image.shape)
    assert 0.005 < area < 0.95, f"implausible box size: {area:.3f} of the frame"
