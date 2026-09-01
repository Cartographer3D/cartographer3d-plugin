from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams
from cartographer.macros.probe_method_wrapper import ProbeMethodWrapperMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from cartographer.probe.probe import Probe


def _make_params(**kwargs: str) -> MockParams:
    p = MockParams()
    p.params.update(kwargs)
    return p


def _make_fallback() -> Mock:
    fallback = Mock(spec=Macro)
    fallback.description = None
    return fallback


def _make_gcode() -> GCodeDispatch:
    def clone_params(params: MacroParams, overrides: dict[str, str]) -> MacroParams:
        new_p = MockParams()
        new_p.params.update(params.get_command_parameters())
        new_p.params.update(overrides)
        return new_p

    gcode = Mock(spec=GCodeDispatch)
    gcode.clone_params = clone_params
    return gcode


@pytest.fixture
def fallback() -> Mock:
    return _make_fallback()


@pytest.fixture
def mesh_fallback() -> Mock:
    return _make_fallback()


@pytest.fixture
def wrapper(probe: Probe, fallback: Mock) -> ProbeMethodWrapperMacro:
    macro = ProbeMethodWrapperMacro(probe, _make_gcode(), "Z_TILT_ADJUST")
    macro.set_fallback_macro(fallback)
    return macro


@pytest.fixture
def mesh_wrapper(probe: Probe, mesh_fallback: Mock) -> ProbeMethodWrapperMacro:
    macro = ProbeMethodWrapperMacro(probe, _make_gcode(), "BED_MESH_CALIBRATE")
    macro.set_fallback_macro(mesh_fallback)
    return macro


class TestClassAttributes:
    def test_missing_fallback_raises(self, probe: Probe) -> None:
        macro = ProbeMethodWrapperMacro(probe, _make_gcode(), "Z_TILT_ADJUST")
        with pytest.raises(RuntimeError, match="Fallback for ProbeMethodWrapperMacro not found"):
            macro.run(_make_params())


class TestScanMode:
    @pytest.mark.parametrize("probe_method", [None, "scan", "SCAN"])
    def test_scan_delegates_original_params_unchanged(
        self, probe_method: str | None, wrapper: ProbeMethodWrapperMacro, fallback: Mock
    ) -> None:
        params = _make_params(**({"PROBE_METHOD": probe_method} if probe_method is not None else {}))
        wrapper.run(params)
        fallback.run.assert_called_once_with(params)

    def test_mesh_scan_defaults_delegate_unchanged(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params()
        mesh_wrapper.run(params)
        mesh_fallback.run.assert_called_once_with(params)


class TestUnknownProbeMethod:
    def test_unknown_method_raises_before_fallback(self, wrapper: ProbeMethodWrapperMacro, fallback: Mock) -> None:
        params = _make_params(PROBE_METHOD="laser")
        with pytest.raises(ValueError, match="Unknown PROBE_METHOD"):
            wrapper.run(params)
        fallback.run.assert_not_called()


class TestTouchModeNonMesh:
    def test_touch_delegates_original_params_inside_touch_context(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mode_during_fallback: list[object] = []

        def capture_mode(p: MacroParams) -> None:
            del p
            mode_during_fallback.append(probe.current_mode)

        fallback.run.side_effect = capture_mode
        original_mode = probe.current_mode
        wrapper.run(params)

        fallback.run.assert_called_once_with(params)
        assert mode_during_fallback[0] is probe.touch
        assert probe.current_mode is original_mode

    def test_touch_restores_mode_on_exception(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        fallback.run.side_effect = RuntimeError("levelling failed")
        original_mode = probe.current_mode
        with pytest.raises(RuntimeError, match="levelling failed"):
            wrapper.run(params)
        assert probe.current_mode is original_mode


class TestTouchModeMesh:
    def test_method_absent_defaults_to_automatic(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_wrapper.run(params)
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("METHOD", None) == "automatic"

    @pytest.mark.parametrize("method", ["automatic", "AUTOMATIC"])
    def test_method_automatic_accepted(
        self, method: str, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch", METHOD=method)
        mesh_wrapper.run(params)
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("METHOD", None) == "automatic"

    def test_non_automatic_method_raises(self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock) -> None:
        params = _make_params(PROBE_METHOD="touch", METHOD="scan")
        with pytest.raises(ValueError, match="METHOD=automatic"):
            mesh_wrapper.run(params)
        mesh_fallback.run.assert_not_called()

    def test_default_bounds_from_touch_boundaries(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_wrapper.run(params)
        b = probe.touch.boundaries
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("MESH_MIN", None) == f"{b.min_x},{b.min_y}"
        assert called_params.get("MESH_MAX", None) == f"{b.max_x},{b.max_y}"

    def test_explicit_bounds_preserved(self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock) -> None:
        params = _make_params(PROBE_METHOD="touch", MESH_MIN="10,10", MESH_MAX="190,190")
        mesh_wrapper.run(params)
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("MESH_MIN", None) == "10,10"
        assert called_params.get("MESH_MAX", None) == "190,190"

    def test_touch_context_active_during_fallback(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mode_during_fallback: list[object] = []

        def capture_mode(p: MacroParams) -> None:
            del p
            mode_during_fallback.append(probe.current_mode)

        mesh_fallback.run.side_effect = capture_mode
        original_mode = probe.current_mode
        mesh_wrapper.run(params)

        assert mode_during_fallback[0] is probe.touch
        assert probe.current_mode is original_mode

    def test_touch_restores_mode_on_exception(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_fallback.run.side_effect = RuntimeError("mesh failed")
        original_mode = probe.current_mode
        with pytest.raises(RuntimeError, match="mesh failed"):
            mesh_wrapper.run(params)
        assert probe.current_mode is original_mode
