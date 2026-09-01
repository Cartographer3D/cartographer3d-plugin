from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams
from cartographer.macros.probe_method_wrapper import ProbeMethodWrapperMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from pytest import FixtureRequest

    from cartographer.probe.probe import Probe


def _make_params(**kwargs: str) -> MockParams:
    p = MockParams()
    p.params.update(kwargs)
    return p


def _make_gcode() -> GCodeDispatch:
    def clone_params(params: MacroParams, overrides: dict[str, str]) -> MacroParams:
        new_p = MockParams()
        new_p.params.update(params.get_command_parameters())
        new_p.params.update(overrides)
        return new_p

    gcode = Mock(spec=GCodeDispatch)
    gcode.clone_params = clone_params
    return gcode


def _make_wrapper(probe: Probe, command_name: str) -> tuple[ProbeMethodWrapperMacro, Mock]:
    fallback = Mock(spec=Macro)
    fallback.description = None
    macro = ProbeMethodWrapperMacro(probe, _make_gcode(), command_name)
    macro.set_fallback_macro(fallback)
    return macro, fallback


@pytest.fixture
def wrapper(probe: Probe) -> tuple[ProbeMethodWrapperMacro, Mock]:
    return _make_wrapper(probe, "Z_TILT_ADJUST")


@pytest.fixture
def mesh_wrapper(probe: Probe) -> tuple[ProbeMethodWrapperMacro, Mock]:
    return _make_wrapper(probe, "BED_MESH_CALIBRATE")


def test_missing_fallback_raises(probe: Probe) -> None:
    macro = ProbeMethodWrapperMacro(probe, _make_gcode(), "Z_TILT_ADJUST")
    with pytest.raises(RuntimeError, match="Fallback for ProbeMethodWrapperMacro not found"):
        macro.run(_make_params())


@pytest.mark.parametrize("probe_method", [None, "scan", "SCAN"])
def test_scan_delegates_original_params_unchanged(
    probe_method: str | None, wrapper: tuple[ProbeMethodWrapperMacro, Mock]
) -> None:
    macro, fallback = wrapper
    params = _make_params(**({"PROBE_METHOD": probe_method} if probe_method is not None else {}))
    macro.run(params)
    fallback.run.assert_called_once_with(params)


def test_mesh_scan_defaults_delegate_unchanged(mesh_wrapper: tuple[ProbeMethodWrapperMacro, Mock]) -> None:
    macro, fallback = mesh_wrapper
    params = _make_params()
    macro.run(params)
    fallback.run.assert_called_once_with(params)


def test_unknown_method_raises_before_fallback(wrapper: tuple[ProbeMethodWrapperMacro, Mock]) -> None:
    macro, fallback = wrapper
    params = _make_params(PROBE_METHOD="laser")
    with pytest.raises(ValueError, match="Unknown PROBE_METHOD"):
        macro.run(params)
    fallback.run.assert_not_called()


@pytest.mark.parametrize("wrapper_name", ["wrapper", "mesh_wrapper"])
@pytest.mark.parametrize("raises", [False, True])
def test_touch_context_active_and_restored(
    raises: bool, wrapper_name: str, probe: Probe, request: FixtureRequest
) -> None:
    macro, fallback = request.getfixturevalue(wrapper_name)
    params = _make_params(PROBE_METHOD="touch")
    mode_during_fallback: list[object] = []

    def capture_mode(p: MacroParams) -> None:
        del p
        mode_during_fallback.append(probe.current_mode)
        if raises:
            msg = "failed"
            raise RuntimeError(msg)

    fallback.run.side_effect = capture_mode
    original_mode = probe.current_mode

    if raises:
        with pytest.raises(RuntimeError, match="failed"):
            macro.run(params)
    else:
        macro.run(params)
        if wrapper_name == "wrapper":
            fallback.run.assert_called_once_with(params)
        else:
            fallback.run.assert_called_once()

    assert mode_during_fallback[0] is probe.touch
    assert probe.current_mode is original_mode


@pytest.mark.parametrize(
    ("method", "expected_method", "expect_error"),
    [
        (None, "automatic", False),
        ("automatic", "automatic", False),
        ("AUTOMATIC", "automatic", False),
        ("scan", None, True),
    ],
)
def test_mesh_method_normalization(
    method: str | None,
    expected_method: str | None,
    expect_error: bool,
    mesh_wrapper: tuple[ProbeMethodWrapperMacro, Mock],
) -> None:
    macro, fallback = mesh_wrapper
    params = _make_params(PROBE_METHOD="touch", **({"METHOD": method} if method is not None else {}))
    if expect_error:
        with pytest.raises(ValueError, match="METHOD=automatic"):
            macro.run(params)
        fallback.run.assert_not_called()
        return
    macro.run(params)
    called_params: MacroParams = fallback.run.call_args[0][0]
    assert called_params.get("METHOD", None) == expected_method


@pytest.mark.parametrize(("mesh_min", "mesh_max"), [(None, None), ("10,10", "190,190")])
def test_mesh_bounds(
    mesh_min: str | None,
    mesh_max: str | None,
    mesh_wrapper: tuple[ProbeMethodWrapperMacro, Mock],
    probe: Probe,
) -> None:
    macro, fallback = mesh_wrapper
    if mesh_min is None or mesh_max is None:
        overrides: dict[str, str] = {}
    else:
        overrides = {"MESH_MIN": mesh_min, "MESH_MAX": mesh_max}
    params = _make_params(PROBE_METHOD="touch", **overrides)
    macro.run(params)
    called_params: MacroParams = fallback.run.call_args[0][0]
    if mesh_min is None:
        b = probe.touch.boundaries
        assert called_params.get("MESH_MIN", None) == f"{b.min_x},{b.min_y}"
        assert called_params.get("MESH_MAX", None) == f"{b.max_x},{b.max_y}"
    else:
        assert called_params.get("MESH_MIN", None) == mesh_min
        assert called_params.get("MESH_MAX", None) == mesh_max
