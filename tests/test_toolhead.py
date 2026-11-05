from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.interfaces.printer import Position, ProbeMode, Toolhead
from cartographer.macros.touch.calibrate import (
    CalibrationProbe,
    CalibrationTouchMode,
    ThresholdScreener,
    ThresholdVerifier,
)
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode
from cartographer.probe.touch_mode import TouchMode
from cartographer.toolhead import BacklashCompensatingToolhead

if TYPE_CHECKING:
    from collections.abc import Callable

    from cartographer.interfaces.printer import MacroParams


@pytest.mark.parametrize(
    "method",
    [
        ProbeMode.perform_probe,
        Probe.perform_scan,
        Probe.perform_touch,
        ScanMode.perform_probe,
        TouchMode.perform_probe,
        Toolhead.z_probing_move,
        Toolhead.z_home_end,
        Toolhead.set_max_accel,
        Toolhead.manual_probe,
        BacklashCompensatingToolhead.z_probing_move,
        BacklashCompensatingToolhead.z_home_end,
        BacklashCompensatingToolhead.set_max_accel,
        BacklashCompensatingToolhead.manual_probe,
        CalibrationProbe.collect_samples,
        CalibrationProbe.perform_touch_probe,
        CalibrationTouchMode.collect_samples,
        CalibrationTouchMode.perform_touch_probe,
        ThresholdScreener.screen,
        ThresholdVerifier.verify,
        *(getattr(TouchMode, name) for name in ("_run_probe", "_perform_single_probe")),
    ],
)
def test_physical_operations_require_explicit_command(method: Callable[..., object]) -> None:
    parameter = inspect.signature(method).parameters["gcmd"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_forwarding_toolhead_preserves_command(params: MacroParams) -> None:
    backend = Mock(spec=Toolhead)
    backend.z_probing_move.return_value = 1.25
    toolhead = BacklashCompensatingToolhead(backend, backlash=0.1)
    endstop = Mock()

    assert toolhead.z_probing_move(endstop, speed=2, gcmd=params) == 1.25
    toolhead.z_home_end(endstop, gcmd=params)
    toolhead.set_max_accel(100, gcmd=params)

    backend.z_probing_move.assert_called_once_with(endstop, speed=2, gcmd=params)
    backend.z_home_end.assert_called_once_with(endstop, gcmd=params)
    backend.set_max_accel.assert_called_once_with(100, gcmd=params)
    for method in (backend.z_probing_move, backend.z_home_end, backend.set_max_accel):
        assert method.call_args.kwargs["gcmd"] is params


@pytest.mark.parametrize("position", [Position(1, 2, 3), None])
def test_forwarding_toolhead_preserves_manual_callback(params: MacroParams, position: Position | None) -> None:
    backend = Mock(spec=Toolhead)
    toolhead = BacklashCompensatingToolhead(backend, backlash=0.1)
    callback = Mock()
    completing = Mock()

    toolhead.manual_probe(callback, gcmd=params)

    backend.manual_probe.assert_called_once_with(callback, gcmd=params)
    assert backend.manual_probe.call_args.args[0] is callback
    assert backend.manual_probe.call_args.kwargs["gcmd"] is params
    backend.manual_probe.call_args.args[0](position, completing)
    callback.assert_called_once_with(position, completing)
    assert callback.call_args.args[1] is completing
