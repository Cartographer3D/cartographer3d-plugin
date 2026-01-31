from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

from cartographer.macros.probe import ProbeMacro

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture

    from cartographer.interfaces.printer import MacroParams, Toolhead
    from cartographer.probe import Probe


def test_probe_macro_logs_result(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    params: MacroParams,
    toolhead: Toolhead,
):
    """Test PROBE macro logs the probed Z value."""
    probe.scan.perform_probe = mocker.Mock(return_value=2.0)

    macro = ProbeMacro(probe, toolhead)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "Result: at 10.000,10.000 estimate contact at z=0.000000" in caplog.messages


def test_probe_accuracy_respects_samples_parameter(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    """Test PROBE_ACCURACY macro probes the specified number of samples."""
    from typing import cast

    from cartographer.macros.probe import ProbeAccuracyMacro

    # Setup: probe measures 10, samples parameter is 2
    probe.scan.perform_probe = mocker.Mock(return_value=10.0)
    params.get_int = mocker.Mock(return_value=2)

    # Execute
    macro = ProbeAccuracyMacro(probe, toolhead)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: it should probe 2 times
    assert cast("Mock", probe.scan.perform_probe).call_count == 2


def test_probe_accuracy_logs_statistics(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    """Test PROBE_ACCURACY macro logs correct statistics."""
    from cartographer.macros.probe import ProbeAccuracyMacro

    # Setup: probe measures 10 and 20 alternating
    probe.scan.perform_probe = mocker.Mock(side_effect=itertools.cycle([10.0, 20.0]))

    # Execute
    macro = ProbeAccuracyMacro(probe, toolhead)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify: should log statistics
    assert "probe accuracy results" in caplog.text
    assert "minimum 10" in caplog.text
    assert "maximum 20" in caplog.text
    assert "range 10" in caplog.text
    assert "median 15" in caplog.text
    assert "standard deviation 5" in caplog.text


def test_query_probe_when_triggered(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    params: MacroParams,
):
    """Test QUERY_PROBE macro when probe is triggered."""
    from cartographer.macros.probe import QueryProbeMacro

    # Setup: probe is triggered
    probe.scan.query_is_triggered = mocker.Mock(return_value=True)

    # Execute
    macro = QueryProbeMacro(probe)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify
    assert "probe: TRIGGERED" in caplog.text


def test_query_probe_when_not_triggered(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    params: MacroParams,
):
    """Test QUERY_PROBE macro when probe is not triggered."""
    from cartographer.macros.probe import QueryProbeMacro

    # Setup: probe is not triggered
    probe.scan.query_is_triggered = mocker.Mock(return_value=False)

    # Execute
    macro = QueryProbeMacro(probe)
    with caplog.at_level(logging.INFO):
        macro.run(params)

    # Verify
    assert "probe: open" in caplog.text
