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

import cv2
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


def box_area_fraction(box: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Fraction of the frame a box covers.

    A sanity check with teeth: a "detection" covering 90% of the frame usually
    means the model grabbed the whole scene, and a detection covering 0.1% is
    usually a piece of background. Both pass a confidence threshold happily.
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height))


def box_centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)


def box_diagonal(box: np.ndarray) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


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


def draw_detections(
    image: np.ndarray,
    detections: Detection | list[Detection],
    caption: str = "",
    highlight_best: bool = True,
) -> np.ndarray:
    """Draw boxes on a copy of the image.

    This is the artifact that decides whether stage 1 passes. Metrics can look
    healthy while the box sits on a dump truck, a shadow or the sky, so the
    frames must be looked at, not just scored.
    """
    if isinstance(detections, Detection):
        detections = [detections]

    canvas = image.copy()
    height = canvas.shape[0]
    thickness = max(1, round(height / 400))
    font_scale = max(0.4, height / 1000)

    for detection in detections:
        best = detection.best()
        for i, (box, score) in enumerate(zip(detection.boxes, detection.scores, strict=True)):
            is_best = highlight_best and best is not None and np.allclose(box, best[0])
            colour = _COLOR_PRIMARY if is_best else _COLOR_SECONDARY
            x1, y1, x2, y2 = (round(v) for v in box)
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), colour, thickness + (1 if is_best else 0)
            )

            label = detection.labels[i] if i < len(detection.labels) else detection.prompt
            _draw_label(
                canvas, f"{label} {score:.2f}", (x1, y1), colour, font_scale, thickness
            )

    if caption:
        _draw_label(
            canvas, caption, (6, 6), (40, 40, 40), font_scale, thickness, anchor_below=True
        )
    return canvas


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    font_scale: float,
    thickness: int,
    anchor_below: bool = False,
) -> None:
    """Text on a filled background, so it stays readable over any imagery."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    top = y if anchor_below else max(0, y - text_height - baseline - 4)
    cv2.rectangle(
        canvas,
        (x, top),
        (x + text_width + 6, top + text_height + baseline + 4),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        canvas,
        text,
        (x + 3, top + text_height + 2),
        font,
        font_scale,
        _COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def contact_sheet(
    images: list[np.ndarray], columns: int = 5, tile_width: int = 380
) -> np.ndarray:
    """Tile frames into one image.

    Twenty separate files require twenty clicks; one sheet shows the whole spike
    at a glance, and failures cluster visibly -- e.g. every frame where the arm
    is extended, or everything after the truck arrives.
    """
    if not images:
        raise ValueError("no images to tile")

    columns = max(1, min(columns, len(images)))
    scaled = []
    for image in images:
        height, width = image.shape[:2]
        tile_height = round(height * tile_width / width)
        scaled.append(
            cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        )

    tile_height = max(tile.shape[0] for tile in scaled)
    rows = []
    for start in range(0, len(scaled), columns):
        row_tiles = scaled[start : start + columns]
        padded = [
            cv2.copyMakeBorder(
                tile,
                0,
                tile_height - tile.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )
            for tile in row_tiles
        ]
        while len(padded) < columns:  # pad the final row so hstack works
            padded.append(np.full_like(padded[0], 20))
        rows.append(np.hstack(padded))
    return np.vstack(rows)
