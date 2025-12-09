from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

from cartographer.config import model_validator
from cartographer.config.model_validator import meets_minimum_version
from cartographer.interfaces.configuration import (
    ModelVersionInfo,
    ScanModelConfiguration,
    TouchModelConfiguration,
)

if TYPE_CHECKING:
    from tests.mocks.config import MockConfiguration

validate_and_remove_incompatible_models = partial(
    model_validator.validate_and_remove_incompatible_models,
    minimum_scan_version=(1, 0, 0),
    minimum_touch_version=(1, 1, 0),
)


@pytest.mark.parametrize(
    "version,minimum,expected",
    [
        # Exact match
        ("1.0.0", (1, 0, 0), True),
        ("1.1.0", (1, 1, 0), True),
        # Higher versions
        ("2.0.0", (1, 0, 0), True),
        ("1.1.0", (1, 0, 0), True),
        ("1.0.1", (1, 0, 0), True),
        ("1.2.0", (1, 1, 0), True),
        # Lower versions
        ("0.9.0", (1, 0, 0), False),
        ("1.0.0", (1, 1, 0), False),
        ("1.0.9", (1, 1, 0), False),
    ],
)
def test_stable_versions(version: str, minimum: tuple[int, int, int], expected: bool):
    assert meets_minimum_version(version, minimum) == expected


@pytest.mark.parametrize(
    "version,minimum,expected",
    [
        # Prereleases of previous version should fail
        ("1.0.0a1", (2, 0, 0), False),
        ("1.0.0b1", (1, 1, 0), False),
        ("1.0.0rc1", (1, 0, 1), False),
        ("1.1.0a1", (1, 2, 0), False),
        # Prereleases of exact version should pass
        ("1.0.0a1", (1, 0, 0), True),
        ("1.0.0b1", (1, 0, 0), True),
        ("1.0.0rc1", (1, 0, 0), True),
        ("1.1.0a1", (1, 1, 0), True),
        # Prereleases of higher versions should pass
        ("1.1.0a1", (1, 0, 0), True),
        ("2.0.0a1", (1, 0, 0), True),
        ("1.2.0rc1", (1, 1, 0), True),
    ],
)
def test_prerelease_versions(version: str, minimum: tuple[int, int, int], expected: bool):
    assert meets_minimum_version(version, minimum) == expected


@pytest.mark.parametrize(
    "version",
    [
        "",
        "invalid",
        "abc.def.ghi",
        "v1.0.0",  # Leading 'v' not supported
    ],
)
def test_invalid_versions(version: str):
    # Invalid versions should return False (except 1.0.0.0 which matches)
    assert meets_minimum_version(version, (1, 0, 0)) is False


def test_removes_scan_model_with_mcu_mismatch(config: MockConfiguration):
    config.scan.models["test"] = ScanModelConfiguration(
        name="test",
        coefficients=[1.0, 2.0, 3.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version="old_mcu", software_version="1.1.0"),
    )

    validate_and_remove_incompatible_models(config, "new_mcu")

    assert "test" not in config.scan.models
    assert any("Removing incompatible scan model" in w for w in config.runtime_warnings)


def test_removes_touch_model_with_mcu_mismatch(config: MockConfiguration):
    config.touch.models["test"] = TouchModelConfiguration(
        name="test",
        threshold=1000,
        speed=3.0,
        z_offset=0.0,
        version_info=ModelVersionInfo(mcu_version="old_mcu", software_version="1.2.0"),
    )

    validate_and_remove_incompatible_models(config, "new_mcu")

    assert "test" not in config.touch.models
    assert any("Removing incompatible touch model" in w for w in config.runtime_warnings)


def test_removes_model_with_old_software_version(config: MockConfiguration):
    config.scan.models["test"] = ScanModelConfiguration(
        name="test",
        coefficients=[1.0, 2.0, 3.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version="abc123", software_version="0.5.0"),
    )

    validate_and_remove_incompatible_models(config, "abc123")

    assert "test" not in config.scan.models
    assert any("software version too old" in w for w in config.runtime_warnings)


def test_keeps_compatible_model(config: MockConfiguration):
    config.scan.models["test"] = ScanModelConfiguration(
        name="test",
        coefficients=[1.0, 2.0, 3.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version="abc123", software_version="1.1.0"),
    )

    validate_and_remove_incompatible_models(config, "abc123")

    assert "test" in config.scan.models
    assert not any("Removing" in w for w in config.runtime_warnings)


def test_warns_about_old_model_without_version_tracking(config: MockConfiguration):
    config.scan.models["legacy"] = ScanModelConfiguration(
        name="legacy",
        coefficients=[1.0, 2.0, 3.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version=None, software_version="1.2.0"),
    )

    validate_and_remove_incompatible_models(config, "abc123")

    # Model should be kept but warning issued
    assert "legacy" in config.scan.models
    assert any("Old scan model" in w and "Consider recalibrating" in w for w in config.runtime_warnings)


def test_validates_multiple_models(config: MockConfiguration):
    # Add multiple scan models
    config.scan.models["good"] = ScanModelConfiguration(
        name="good",
        coefficients=[1.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version="abc123", software_version="1.1.0"),
    )
    config.scan.models["bad"] = ScanModelConfiguration(
        name="bad",
        coefficients=[1.0],
        domain=(0.0, 100.0),
        z_offset=0.0,
        reference_temperature=25.0,
        version_info=ModelVersionInfo(mcu_version="wrong_mcu", software_version="1.1.0"),
    )

    # Add multiple touch models
    config.touch.models["good_touch"] = TouchModelConfiguration(
        name="good_touch",
        threshold=1000,
        speed=3.0,
        z_offset=0.0,
        version_info=ModelVersionInfo(mcu_version="abc123", software_version="1.2.0"),
    )
    config.touch.models["bad_touch"] = TouchModelConfiguration(
        name="bad_touch",
        threshold=1000,
        speed=3.0,
        z_offset=0.0,
        version_info=ModelVersionInfo(mcu_version="abc123", software_version="0.5.0"),
    )

    validate_and_remove_incompatible_models(config, "abc123")

    assert "good" in config.scan.models
    assert "bad" not in config.scan.models
    assert "good_touch" in config.touch.models
    assert "bad_touch" not in config.touch.models
