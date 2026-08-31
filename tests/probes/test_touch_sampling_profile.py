"""Tests for model-specific sampling profile (samples / sample_range) with global fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.configuration import Configuration, TouchModelConfiguration
from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


# ---------------------------------------------------------------------------
# Fixtures — use project-level conftest fixtures (probe, toolhead, config)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def base_model(probe: Probe, config: Configuration) -> None:
    """Register a minimal model with no sampling overrides (legacy-style)."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="default_model",
            speed=3,
            threshold=1000,
            z_offset=0,
            # samples and sample_range intentionally absent → None
        )
    )
    probe.touch.load_model("default_model")


# ---------------------------------------------------------------------------
# Legacy fallback: model without samples/sample_range uses global config
# ---------------------------------------------------------------------------


def test_legacy_model_uses_global_samples(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    """A model with samples=None falls back to global samples (5 from MockConfiguration)."""
    # Global config has samples=5; supply exactly 5 agreeing touches after 2 noisy ones
    # (window = 5 + 2 = 7; first 2 noisy, then 5 consistent → success)
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    result = probe.touch.perform_probe()
    assert result == 0.5


def test_legacy_model_uses_global_sample_range(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    """A model with sample_range=None falls back to global sample_range (0.010)."""
    # 10 monotone values, each 0.003 apart: best-5-in-window-7 = range 0.012 > 0.010
    side = [round(0.500 + i * 0.003, 4) for i in range(10)]
    toolhead.z_probing_move = mocker.Mock(side_effect=side)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


# ---------------------------------------------------------------------------
# Model-specific overrides: stored values override global config
# ---------------------------------------------------------------------------


def test_model_samples_override_used(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: Configuration
) -> None:
    """A model with samples=3 requires only 3 agreeing samples, even though global is 5."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="tight_samples",
            speed=3,
            threshold=1000,
            z_offset=0,
            samples=3,
        )
    )
    probe.touch.load_model("tight_samples")

    # 3 agreeing samples succeeds immediately (global would require 5)
    toolhead.z_probing_move = mocker.Mock(side_effect=[0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    result = probe.touch.perform_probe()
    assert result == 0.5


def test_model_sample_range_override_used(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: Configuration
) -> None:
    """A model with sample_range=0.005 rejects a spread that global 0.010 would accept."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="tight_range",
            speed=3,
            threshold=1000,
            z_offset=0,
            sample_range=0.005,
        )
    )
    probe.touch.load_model("tight_range")

    # 10 monotone values 0.002 apart; best-5-in-window-7 spans 0.008
    # 0.008 < global 0.010 but > model 0.005 → should fail
    side = [round(0.500 + i * 0.002, 4) for i in range(10)]
    toolhead.z_probing_move = mocker.Mock(side_effect=side)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_model_sample_range_override_accepts_tight_spread(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: Configuration
) -> None:
    """A model with sample_range=0.015 accepts a spread that global 0.010 would reject."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="loose_range",
            speed=3,
            threshold=1000,
            z_offset=0,
            sample_range=0.015,
        )
    )
    probe.touch.load_model("loose_range")

    # 5 samples spanning 0.012 — exceeds global 0.010, accepted by model's 0.015
    toolhead.z_probing_move = mocker.Mock(side_effect=[0.500, 0.504, 0.508, 0.510, 0.512])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    result = probe.touch.perform_probe()
    assert abs(result - 0.508) < 0.001  # median of the 5 values


# ---------------------------------------------------------------------------
# Sliding window with model samples override — anti-cherry-picking preserved
# ---------------------------------------------------------------------------


def test_model_samples_window_anti_cherry_picking(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: Configuration
) -> None:
    """With model samples=3 and global max_noisy_samples=2, window=5.
    Spread-out good samples beyond the window (every 3rd position) must still be rejected
    because no window of 5 can contain all 3 agreeing samples without noisy ones."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="window_check",
            speed=3,
            threshold=1000,
            z_offset=0,
            samples=3,
        )
    )
    probe.touch.load_model("window_check")

    # Good values (0.5) appear every 3rd position; noises are monotone 2.0-2.5.
    # Every 5-element window contains at most 2 goods and always has noise pairs
    # that span >0.010 — no valid 3-sample subset exists in any window.
    side = [0.5, 2.0, 2.1, 0.5, 2.2, 2.3, 0.5, 2.4, 2.5, 0.5]
    toolhead.z_probing_move = mocker.Mock(side_effect=side)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_model_samples_window_succeeds_consecutive(
    mocker: MockerFixture, toolhead: Toolhead, probe: Probe, config: Configuration
) -> None:
    """With model samples=3 and global max_noisy_samples=2, window=5.
    Three consecutive good samples within the window succeed."""
    config.save_touch_model(
        TouchModelConfiguration(
            name="window_consecutive",
            speed=3,
            threshold=1000,
            z_offset=0,
            samples=3,
        )
    )
    probe.touch.load_model("window_consecutive")

    # 2 noisy then 3 consistent
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 2.0, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    result = probe.touch.perform_probe()
    assert result == 0.5
