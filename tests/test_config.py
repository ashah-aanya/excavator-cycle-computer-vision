"""Tests for configuration loading.

These are cheap but they protect something that matters: a mistyped key in a
YAML file must fail loudly, not silently leave a default in place and send us
hunting for why a threshold "didn't do anything".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from excavator_cycles.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_defaults_load():
    config = Config.load()
    assert config.sampling.rate_hz == 10.0
    assert config.fsm.hold_seconds == 0.30


def test_shipped_yaml_matches_code_defaults():
    """configs/default.yaml should document the defaults, not diverge from them."""
    from_code = Config.load()
    from_yaml = Config.load(CONFIG_PATH)
    assert from_yaml == from_code


def test_overrides_win():
    config = Config.load(CONFIG_PATH, overrides={"sampling": {"rate_hz": 15.0}})
    assert config.sampling.rate_hz == 15.0
    # A partial override must leave its siblings alone.
    assert config.sampling.anchor_rate_hz == 1.0


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(overrides={"sampling": {"rate_hzz": 10.0}})


def test_config_is_serialisable():
    """Run metadata records the config, so every result is traceable."""
    data = Config.load().to_dict()
    assert data["fsm"]["hold_seconds"] == 0.30
