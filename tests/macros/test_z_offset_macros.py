from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.configuration import ScanModelConfiguration, TouchModelConfiguration
from cartographer.interfaces.printer import Position, Sample
from cartographer.macros.touch import TouchHomeMacro

if TYPE_CHECKING:
    from pytest import LogCaptureFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import MacroParams, Toolhead
    from cartographer.probe import Probe
    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode
    from cartographer.stream import Session


def sample(frequency: float):
    """Helper to create a Sample with specific frequency."""
    return Sample(
        raw_count=0,
        frequency=1 / frequency,
        time=0,
        position=Position(0, 0, frequency),
        temperature=0,
    )


@pytest.fixture
def calibrated_scan(scan: ScanMode, config: Configuration, session: Session[Sample]):
    """Fixture to setup a calibrated scan mode."""
    config.save_scan_model(
        ScanModelConfiguration(
            name="default",
            coefficients=[0.3] * 9,
            domain=(0.1, 5.5),
            z_offset=0.0,
            reference_temperature=30,
        )
    )
    scan.load_model("default")
    session.get_items = lambda: [sample(frequency=2) for _ in range(11)]
    return scan


@pytest.fixture
def calibrated_touch(touch: TouchMode, config: Configuration):
    """Fixture to setup a calibrated touch mode."""
    config.touch.models["default"] = TouchModelConfiguration(
        name="default",
        threshold=1000,
        speed=3,
        z_offset=0.0,
    )
    touch.load_model("default")
    return touch


def test_z_offset_apply_scan_nozzle_raised(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_scan: ScanMode,
):
    """Test Z_OFFSET_APPLY_PROBE with scan mode when nozzle is raised 0.4mm."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: scan z-offset is 2.0, G28 was run, nozzle was raised 0.4mm
    config.save_scan_model(replace(config.scan.models["default"], z_offset=2.0))
    calibrated_scan.load_model("default")
    toolhead.z_home_end(probe.scan)
    toolhead.get_gcode_z_offset = lambda: 0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set scan z-offset to 1.6
    assert config.scan.models["default"].z_offset == 1.6


def test_z_offset_apply_scan_nozzle_lowered(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_scan: ScanMode,
):
    """Test Z_OFFSET_APPLY_PROBE with scan mode when nozzle is lowered 0.4mm."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: scan z-offset is 2.0, G28 was run, nozzle was lowered 0.4mm
    config.save_scan_model(replace(config.scan.models["default"], z_offset=2.0))
    calibrated_scan.load_model("default")
    toolhead.z_home_end(probe.scan)
    toolhead.get_gcode_z_offset = lambda: -0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set scan z-offset to 2.4
    assert config.scan.models["default"].z_offset == 2.4


def test_z_offset_apply_touch_nozzle_raised(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_touch: TouchMode,
):
    """Test Z_OFFSET_APPLY_PROBE with touch mode when nozzle is raised 0.4mm."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: touch z-offset is 0.0, TOUCH_HOME was run, nozzle was raised 0.4mm
    config.save_touch_model(replace(config.touch.models["default"], z_offset=0.0))
    calibrated_touch.load_model("default")
    touch_home_macro = TouchHomeMacro(
        calibrated_touch, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0
    )
    touch_home_macro.run(params)
    toolhead.get_gcode_z_offset = lambda: 0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set touch z-offset to -0.4
    assert config.touch.models["default"].z_offset == -0.4


def test_z_offset_apply_touch_nozzle_lowered(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_touch: TouchMode,
):
    """Test Z_OFFSET_APPLY_PROBE with touch mode when nozzle is lowered 0.4mm."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: touch z-offset is -0.2, TOUCH_HOME was run, nozzle was lowered 0.4mm
    config.save_touch_model(replace(config.touch.models["default"], z_offset=-0.2))
    calibrated_touch.load_model("default")
    touch_home_macro = TouchHomeMacro(
        calibrated_touch, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0
    )
    touch_home_macro.run(params)
    toolhead.get_gcode_z_offset = lambda: -0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set touch z-offset to 0
    assert config.touch.models["default"].z_offset == 0


def test_z_offset_apply_touch_after_scan(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_scan: ScanMode,
    calibrated_touch: TouchMode,
):
    """Test Z_OFFSET_APPLY_PROBE applies to touch mode when touch was run after scan."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: both modes calibrated with different z-offsets
    config.save_scan_model(replace(config.scan.models["default"], z_offset=2.0))
    calibrated_scan.load_model("default")
    config.save_touch_model(replace(config.touch.models["default"], z_offset=0.0))
    calibrated_touch.load_model("default")

    # G28 was run, then TOUCH_HOME was run, nozzle was raised 0.4mm
    toolhead.z_home_end(probe.scan)
    touch_home_macro = TouchHomeMacro(
        calibrated_touch, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0
    )
    touch_home_macro.run(params)
    toolhead.get_gcode_z_offset = lambda: 0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set touch z-offset to -0.4 (latest homing mode)
    assert config.touch.models["default"].z_offset == -0.4


def test_z_offset_apply_scan_after_touch(
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    params: MacroParams,
    calibrated_scan: ScanMode,
    calibrated_touch: TouchMode,
):
    """Test Z_OFFSET_APPLY_PROBE applies to scan mode when scan was run after touch."""
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    # Setup: both modes calibrated with different z-offsets
    config.save_scan_model(replace(config.scan.models["default"], z_offset=2.0))
    calibrated_scan.load_model("default")
    config.save_touch_model(replace(config.touch.models["default"], z_offset=0.0))
    calibrated_touch.load_model("default")

    # TOUCH_HOME was run, then G28 was run, nozzle was raised 0.4mm
    touch_home_macro = TouchHomeMacro(
        calibrated_touch, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0
    )
    touch_home_macro.run(params)
    toolhead.z_home_end(probe.scan)
    toolhead.get_gcode_z_offset = lambda: 0.4

    # Execute
    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should set scan z-offset to 1.6 (latest homing mode)
    assert config.scan.models["default"].z_offset == 1.6
