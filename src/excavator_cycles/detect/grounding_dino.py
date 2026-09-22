"""Grounding DINO: open-vocabulary detection from a text prompt.

How the model works, briefly, because the quirks below follow from it: a normal
detector ends in a fixed classifier (80 COCO classes, none of them "excavator").
Grounding DINO instead encodes the image and the prompt separately, lets them
attend to each other, and scores each candidate box by *similarity to the words*.
Classification becomes matching, so the vocabulary is whatever you type.

Three consequences that this module handles:

1. **Scores are not comparable across prompts.** A 0.4 for "excavator" and a 0.4
   for "dump truck" are not the same confidence; rare words score lower across
   the board. So one phrase per call, and per-phrase calibration later.

2. **Labels can be cross-attributed** when several phrases share one prompt
   string. We read the returned label strings rather than trusting index order.

3. **Prompt format matters.** Phrases are lower-case and end with a period --
   that is the convention the model was trained with.

Weights come from Hugging Face and are cached locally after the first run
(roughly 700 MB for the base model).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..devices import resolve_device
from ..logging_setup import get_logger
from .base import Detection

log = get_logger(__name__)


class GroundingDinoDetector:
    """Text-prompted object detector.

    Heavy imports happen in ``__init__``, not at module import, so the rest of
    the package stays usable on a machine without the deep-learning extras
    installed.
    """

    name = "grounding_dino"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        revision: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "The perception stage needs the model extras. Install with:\n"
                "    uv sync --extra models"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.revision = revision
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = resolve_device(device)

        log.info("loading %s (revision=%s) on %s", model_id, revision or "main", self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, revision=revision
        ).to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        """Find ``prompt`` in a BGR image.

        Returns an empty Detection when nothing clears the threshold. "Not
        found" is data, not an error: a frame where the machine is genuinely
        occluded should lower the coverage metric, not stop the run.
        """
        import torch

        # OpenCV gives BGR; the processor expects RGB. cvtColor (not a
        # reversed slice) because a negative-stride view cannot be turned
        # into a torch tensor -- it raises deep inside the processor.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = _normalise_prompt(prompt)

        inputs = self.processor(images=rgb, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        height, width = image.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],  # converts normalised boxes to pixels
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        # `text_labels` on recent transformers, `labels` on older ones. Reading
        # the strings (rather than indices) is what protects against phrases
        # being cross-attributed.
        labels = results.get("text_labels", results.get("labels", []))
        labels = [str(label) for label in labels]

        log.debug("prompt %r -> %d box(es), scores=%s", text, len(boxes), np.round(scores, 3))
        return Detection(boxes=boxes, scores=scores, labels=labels, prompt=text)


class Owlv2Detector:
    """OWLv2, the same job with a different architecture.

    Present so the spike can compare two Apache-2.0 open-vocabulary detectors on
    identical frames. Construction machinery appears in no standard benchmark,
    so which one wins here is not predictable from published numbers -- it has
    to be measured on the actual footage.
    """

    name = "owlv2"

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        revision: str | None = None,
        box_threshold: float = 0.2,
        device: str | None = None,
        **_ignored: object,
    ) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "The perception stage needs the model extras. Install with:\n"
                "    uv sync --extra models"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.device = resolve_device(device)

        log.info("loading %s on %s", model_id, self.device)
        self.processor = Owlv2Processor.from_pretrained(model_id, revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, revision=revision).to(
            self.device
        )
        self.model.eval()

    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # OWLv2 takes a list of queries; ours are plain nouns, no trailing period.
        query = _normalise_prompt(prompt).rstrip(".").strip()

        inputs = self.processor(images=rgb, text=[[query]], return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)

        height, width = image.shape[:2]
        target_sizes = torch.tensor([[height, width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=self.box_threshold, target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        return Detection(
            boxes=boxes,
            scores=scores,
            labels=[query] * len(boxes),
            prompt=query,
        )


def _normalise_prompt(prompt: str) -> str:
    """Lower-case, period-terminated: the format the model was trained on."""
    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."
    return text
