"""Detection: finding the excavator in a frame.

`build_detector` is the only thing the rest of the pipeline calls, so swapping
models is a config change rather than a code change. Implementations import
torch lazily, so this package stays importable without the model extras.
"""

from __future__ import annotations

from ..config import DetectionConfig
from .base import (
    Detection,
    Detector,
    iou,
)

__all__ = [
    "Detection",
    "Detector",
    "build_detector",
    "iou",
]


def build_detector(name: str, config: DetectionConfig, device: str | None = None) -> Detector:
    """Construct a detector by name.

    Args:
        name: "grounding_dino" (default) or "owlv2".
        config: thresholds and model ids, from the central config.
        device: override the automatic choice ("cuda", "mps", "cpu").
    """
    from .grounding_dino import GroundingDinoDetector, Owlv2Detector

    if name == "grounding_dino":
        return GroundingDinoDetector(
            model_id=config.model_id,
            revision=config.revision,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            device=device,
        )
    if name == "owlv2":
        return Owlv2Detector(model_id=config.alt_model_id, device=device)
    raise ValueError(f"unknown detector: {name!r} (expected 'grounding_dino' or 'owlv2')")
