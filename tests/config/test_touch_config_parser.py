"""Tests for TouchConfig parsing, specifically the sample_range parameter."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.configuration import TouchConfig

if TYPE_CHECKING:
    from tests.mocks.config import MockConfiguration


def test_default_sample_range_in_touch_config(config: MockConfiguration) -> None:
    """Test that default sample_range is set correctly."""
    assert config.touch.sample_range == 0.010


def test_custom_sample_range_in_touch_config(config: MockConfiguration) -> None:
    """Test that custom sample_range value can be set."""
    custom_touch_config = replace(config.touch, sample_range=0.005)

    assert custom_touch_config.sample_range == 0.005


@pytest.mark.parametrize(
    "sample_range",
    [
        0.001,  # 1 micron (minimum)
        0.005,  # 5 microns
        0.010,  # 10 microns (default)
        0.015,  # 15 microns (maximum)
    ],
)
def test_sample_range_valid_values(config: MockConfiguration, sample_range: float) -> None:
    """Test that various valid sample_range values work correctly."""
    custom_touch_config = replace(config.touch, sample_range=sample_range)

    assert custom_touch_config.sample_range == sample_range
    # Verify the configuration can be used
    assert isinstance(custom_touch_config, TouchConfig)


def test_touch_mode_configuration_includes_sample_range(config: MockConfiguration) -> None:
    """Test that TouchModeConfiguration gets sample_range from config."""
    from cartographer.probe.touch_mode import TouchModeConfiguration
    from tests.mocks.config import MockConfiguration as MockConfig

    # Create a custom config with a specific sample_range
    custom_touch_config = replace(config.touch, sample_range=0.007)
    custom_config = MockConfig(touch=custom_touch_config)

    # Build TouchModeConfiguration from the custom config
    touch_mode_config = TouchModeConfiguration.from_config(custom_config)

    # Verify the sample_range was transferred correctly
    assert touch_mode_config.sample_range == 0.007
