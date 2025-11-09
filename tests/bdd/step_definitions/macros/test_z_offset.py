from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytest_bdd import given, parsers, scenarios, then, when

from cartographer.macros.touch import TouchHomeMacro

if TYPE_CHECKING:
    from pytest import LogCaptureFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import MacroParams, Toolhead
    from cartographer.probe import Probe
    from cartographer.probe.touch_mode import TouchMode
    from tests.bdd.helpers.context import Context


scenarios("../../features/z_offset.feature")


@given(parsers.parse("I have baby stepped the nozzle {offset:g}mm up"))
def given_baby_step_up(toolhead: Toolhead, offset: float):
    toolhead.get_gcode_z_offset = lambda: offset


@given(parsers.parse("I have baby stepped the nozzle {offset:g}mm down"))
def given_baby_step_down(toolhead: Toolhead, offset: float):
    toolhead.get_gcode_z_offset = lambda: -offset


@given("I ran G28")
def given_g28(probe: Probe, toolhead: Toolhead):
    toolhead.z_home_end(probe.scan)


@given("I ran TOUCH_HOME")
def given_touch_home(touch: TouchMode, toolhead: Toolhead, params: MacroParams):
    macro = TouchHomeMacro(touch, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    macro.run(params)


@when("I run the Z_OFFSET_APPLY_PROBE macro")
def when_run_probe_accuracy_macro(
    params: MacroParams,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    config: Configuration,
    context: Context,
):
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    macro = ZOffsetApplyProbeMacro(probe, toolhead, config)
    with caplog.at_level(logging.INFO):
        try:
            macro.run(params)
        except Exception as e:
            context.error = e


@then(parsers.parse("it should set scan z-offset to {offset:g}"))
def then_update_scan_z_offset(config: Configuration, offset: str):
    assert config.scan.models["default"].z_offset == float(offset)


@then(parsers.parse("it should set touch z-offset to {offset:g}"))
def then_update_touch_z_offset(config: Configuration, offset: str):
    assert config.touch.models["default"].z_offset == float(offset)
