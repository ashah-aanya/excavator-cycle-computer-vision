"""Tests for device selection.

The rule under test: an unavailable device is a warning and a fallback, never a
crash. Asking for CUDA on a Mac used to raise "Torch not compiled with CUDA
enabled" from deep inside torch -- a stack trace for what is really a typo.

Availability is injected, so these run identically on a laptop and a GPU node.
"""

from __future__ import annotations

import pytest

from excavator_cycles.devices import best_available, resolve_device

NO = lambda: False
YES = lambda: True


def test_prefers_cuda_then_mps_then_cpu():
    assert best_available(cuda=YES, mps=YES) == "cuda"
    assert best_available(cuda=NO, mps=YES) == "mps"
    assert best_available(cuda=NO, mps=NO) == "cpu"


def test_auto_detects_when_nothing_requested():
    assert resolve_device(None, cuda=NO, mps=YES) == "mps"


def test_cuda_on_a_mac_falls_back_instead_of_crashing(caplog):
    """The exact failure that produced a stack trace on a laptop."""
    with caplog.at_level("WARNING"):
        assert resolve_device("cuda", cuda=NO, mps=YES) == "mps"
    assert "CUDA was requested but is not available" in caplog.text


def test_cuda_honoured_when_present():
    assert resolve_device("cuda", cuda=YES, mps=NO) == "cuda"
    assert resolve_device("cuda:1", cuda=YES, mps=NO) == "cuda:1"


def test_mps_falls_back_on_a_linux_box():
    assert resolve_device("mps", cuda=YES, mps=NO) == "cuda"


def test_cpu_is_always_honoured():
    assert resolve_device("cpu", cuda=YES, mps=YES) == "cpu"


@pytest.mark.parametrize("bogus", ["gpu", "tpu", "CUDAA", ""])
def test_unrecognised_device_falls_back(bogus, caplog):
    with caplog.at_level("WARNING"):
        assert resolve_device(bogus, cuda=NO, mps=NO) == "cpu"
    assert "falling back" in caplog.text
