from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, cast

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from cartographer.probe import Probe
from cartographer.probe.scan_mode import ScanMode
from cartographer.probe.touch_mode import TouchMode

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture

    from cartographer.interfaces.printer import MacroParams, Toolhead


scenarios("../../features/probe.feature")


@pytest.fixture
def scan(mocker: MockerFixture):
    return mocker.MagicMock(spec=ScanMode, instance=True, autospec=True)


@pytest.fixture
def touch(mocker: MockerFixture):
    return mocker.MagicMock(spec=TouchMode, instance=True, autospec=True)


@given("a probe", target_fixture="probe")
def given_probe(scan: ScanMode, touch: TouchMode) -> Probe:
    return Probe(scan, touch)


@given("the probe measures:")
def given_probe_measurements(mocker: MockerFixture, scan: ScanMode, datatable: list[list[str]]):
    scan.perform_probe = mocker.Mock(side_effect=itertools.cycle(map(float, itertools.chain.from_iterable(datatable))))


@given("the probe is triggered")
def given_probe_triggered(mocker: MockerFixture, scan: ScanMode):
    scan.query_is_triggered = mocker.Mock(return_value=True)


@given("the probe is not triggered")
def given_probe_not_triggered(mocker: MockerFixture, scan: ScanMode):
    scan.query_is_triggered = mocker.Mock(return_value=False)


@when("I run the PROBE macro")
def when_run_probe_macro(params: MacroParams, caplog: LogCaptureFixture, probe: Probe):
    from cartographer.macros.probe import ProbeMacro

    macro = ProbeMacro(probe)
    with caplog.at_level(logging.INFO):
        macro.run(params)


@when("I run the PROBE_ACCURACY macro")
def when_run_probe_accuracy_macro(params: MacroParams, caplog: LogCaptureFixture, probe: Probe, toolhead: Toolhead):
    from cartographer.macros.probe import ProbeAccuracyMacro

    macro = ProbeAccuracyMacro(probe, toolhead)
    with caplog.at_level(logging.INFO):
        macro.run(params)


@when("I run the QUERY_PROBE macro")
def when_run_query_probe_macro(params: MacroParams, caplog: LogCaptureFixture, probe: Probe):
    from cartographer.macros.probe import QueryProbeMacro

    macro = QueryProbeMacro(probe)
    with caplog.at_level(logging.INFO):
        macro.run(params)


@then(parsers.parse("it should probe {count:d} times"))
def then_probe_count(scan: ScanMode, count: int):
    assert cast("Mock", scan.perform_probe).call_count == count
