"""Choosing where the models run.

One rule: **an unavailable device is a warning, not a crash.** Asking for CUDA on
a Mac, or for a GPU on a node that has none, should degrade to whatever is
available and say so. The pipeline must always produce an answer, and a wrong
``--device`` flag is not a reason to produce none.

Availability is probed through injectable callables so the logic can be tested
without a GPU, and so this module imports torch lazily -- everything outside the
perception stage stays usable without the deep-learning extras installed.
"""

from __future__ import annotations

from collections.abc import Callable

from .logging_setup import get_logger

log = get_logger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


def best_available(
    cuda: Callable[[], bool] = _cuda_available,
    mps: Callable[[], bool] = _mps_available,
) -> str:
    """The fastest device present: CUDA, then Apple Silicon, then CPU."""
    if cuda():
        return "cuda"
    if mps():
        return "mps"
    return "cpu"


def resolve_device(
    requested: str | None = None,
    cuda: Callable[[], bool] = _cuda_available,
    mps: Callable[[], bool] = _mps_available,
) -> str:
    """Honour ``requested`` when possible; otherwise warn and fall back.

    Args:
        requested: "cuda", "cuda:1", "mps", "cpu", or None to auto-detect.

    Returns:
        A device string that actually works on this machine.
    """
    fallback = best_available(cuda, mps)
    if requested is None:
        log.info("no device requested; using %s", fallback)
        return fallback

    wanted = requested.strip().lower()
    if wanted.startswith("cuda") and not cuda():
        log.warning(
            "CUDA was requested but is not available "
            "(on macOS, PyTorch has no CUDA support); falling back to %s",
            fallback,
        )
        return fallback
    if wanted.startswith("mps") and not mps():
        log.warning("MPS was requested but is not available; falling back to %s", fallback)
        return fallback
    if wanted not in {"cpu"} and not wanted.startswith(("cuda", "mps")):
        log.warning("unrecognised device %r; falling back to %s", requested, fallback)
        return fallback
    return requested
