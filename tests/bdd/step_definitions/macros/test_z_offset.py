from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from cartographer.interfaces.printer import Position
from cartographer.probe import Probe
from cartographer.probe.scan_mode import ScanMode
from cartographer.probe.touch_mode import TouchMode

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture

    from cartographer.interfaces.printer import MacroParams, Toolhead


scenarios("../../features/z_offset.feature")


@pytest.fixture
def scan(mocker: MockerFixture):
    mock = mocker.MagicMock(spec=ScanMode, instance=True, autospec=True)
    mock.is_ready = True
    return mock


@pytest.fixture
def touch(mocker: MockerFixture):
    mock = mocker.MagicMock(spec=TouchMode, instance=True, autospec=True)
    mock.is_ready = False
    return mock


@given("a probe", target_fixture="probe")
def given_probe(scan: ScanMode, touch: TouchMode) -> Probe:
    return Probe(scan, touch)


@given("the probe has touch ready")
def given_probe_touch_model_loaded(touch: Mock):
    touch.is_ready = True


@given(parsers.parse("the probe's current z-offset is {offset:g}"))
def given_current_z_offset(scan: Mock, touch: Mock, offset: float):
    scan.offset = Position(0, 0, offset)
    touch.offset = Position(0, 0, offset)


@given(parsers.parse("I have baby stepped the nozzle {offset:g}mm up"))
def given_baby_step_up(toolhead: Toolhead, offset: float):
    toolhead.get_gcode_z_offset = lambda: offset


@given(parsers.parse("I have baby stepped the nozzle {offset:g}mm down"))
def given_baby_step_down(toolhead: Toolhead, offset: float):
    toolhead.get_gcode_z_offset = lambda: -offset


@when("I run the Z_OFFSET_APPLY_PROBE macro")
def when_run_probe_accuracy_macro(params: MacroParams, caplog: LogCaptureFixture, probe: Probe, toolhead: Toolhead):
    from cartographer.macros.probe import ZOffsetApplyProbeMacro

    macro = ZOffsetApplyProbeMacro(probe, toolhead)
    with caplog.at_level(logging.INFO):
        macro.run(params)


@then("it should set z-offset on the scan model")
def then_set_scan_z_offset(scan: ScanMode):
    assert cast("Mock", scan.save_z_offset).call_count == 1


@then("it should not set z-offset on the scan model")
def then_not_set_scan_z_offset(scan: ScanMode):
    assert cast("Mock", scan.save_z_offset).call_count == 0


@then("it should set z-offset on the touch model")
def then_set_touch_z_offset(touch: TouchMode):
    assert cast("Mock", touch.save_z_offset).call_count == 1


@then(parsers.parse("it should set scan z-offset to {offset:g}"))
def then_update_scan_z_offset(mocker: MockerFixture, scan: ScanMode, offset: str):
    assert cast("Mock", scan.save_z_offset).mock_calls == [mocker.call(offset)]


@then(parsers.parse("it should set touch z-offset to {offset:g}"))
def then_update_touch_z_offset(mocker: MockerFixture, touch: TouchMode, offset: str):
    assert cast("Mock", touch.save_z_offset).mock_calls == [mocker.call(offset)]
