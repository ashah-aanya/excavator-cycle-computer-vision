"""The detector interface, and drawing detections so a human can check them.

Everything downstream sees only ``Detection``. No other module knows which model
produced it, which is what makes ``--detector`` a one-line swap and what lets
the spike compare two models by running identical measurement code over both.

Coordinates are **pixels in the coordinate frame of the image that was passed
in**, as ``[x1, y1, x2, y2]``. Models internally use normalised centre-width-height
boxes; converting at the boundary means no other module has to think about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Overlay colours (BGR). Cosmetic only -- they cannot change any reported
# number, so by the project's convention they live here, not in config.py.
_COLOR_PRIMARY = (60, 200, 60)
_COLOR_SECONDARY = (200, 160, 40)
_COLOR_TEXT = (255, 255, 255)


@dataclass(frozen=True)
class Detection:
    """What a detector found in one image for one text prompt.

    Attributes:
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]`` in pixels.
        scores: ``(N,)`` confidence per box. NOT comparable across prompts --
            open-vocabulary scores depend on the words used, so a 0.4 for
            "excavator" and a 0.4 for "dump truck" do not mean the same thing.
        labels: the phrase each box matched.
        prompt: the text this detection answered.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: list[str] = field(default_factory=list)
    prompt: str = ""

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or (self.boxes.size and self.boxes.shape[1] != 4):
            raise ValueError(f"boxes must be (N, 4), got {self.boxes.shape}")
        if len(self.boxes) != len(self.scores):
            raise ValueError("boxes and scores must have the same length")

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def found(self) -> bool:
        return len(self.boxes) > 0

    def best(self) -> tuple[np.ndarray, float] | None:
        """Highest-scoring box, or None.

        The pipeline tracks a single excavator, so after detection we keep one
        box. Choosing by score (rather than, say, size) keeps this honest: if
        the model is unsure, the QA metrics should see that, not a heuristic
        that papers over it.
        """
        if not self.found:
            return None
        index = int(np.argmax(self.scores))
        return self.boxes[index], float(self.scores[index])

    @staticmethod
    def empty(prompt: str = "") -> Detection:
        return Detection(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=[],
            prompt=prompt,
        )


@runtime_checkable
class Detector(Protocol):
    """Anything that can answer "where is <phrase> in this image?"."""

    name: str

    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        """Detect ``prompt`` in a BGR image. Never raises on "nothing found"."""
        ...





def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection over union: how much two boxes agree.

    Used two ways here: comparing two detectors on the same frame, and later
    comparing a fresh detection against the tracker's propagated mask to catch
    drift.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0




