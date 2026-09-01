from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams
from cartographer.macros.probe_method_wrapper import ProbeMethodWrapperMacro
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from cartographer.probe.probe import Probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**kwargs: str) -> MockParams:
    p = MockParams()
    p.params.update(kwargs)
    return p


def _make_fallback() -> Mock:
    fallback = Mock(spec=Macro)
    fallback.description = None
    return fallback


def _make_gcode() -> GCodeDispatch:
    """Return a GCodeDispatch mock whose clone_params merges overrides into a new MockParams."""

    def clone_params(params: MacroParams, overrides: dict[str, str]) -> MacroParams:
        new_p = MockParams()
        new_p.params.update(params.get_command_parameters())
        new_p.params.update(overrides)
        return new_p

    gcode = Mock(spec=GCodeDispatch)
    gcode.clone_params = clone_params
    return gcode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_requires_fallback_is_true(self) -> None:
        assert ProbeMethodWrapperMacro.requires_fallback is True

    def test_description_is_none(self) -> None:
        assert ProbeMethodWrapperMacro.description is None


# ---------------------------------------------------------------------------
# Scan / default delegation (non-mesh command)
# ---------------------------------------------------------------------------


class TestScanMode:
    def test_no_probe_method_defaults_to_scan_and_delegates(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock
    ) -> None:
        params = _make_params()
        wrapper.run(params)
        fallback.run.assert_called_once_with(params)

    def test_explicit_scan_delegates_original_params_unchanged(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="scan")
        wrapper.run(params)
        fallback.run.assert_called_once_with(params)

    def test_probe_method_case_insensitive(self, wrapper: ProbeMethodWrapperMacro, fallback: Mock) -> None:
        params = _make_params(PROBE_METHOD="SCAN")
        wrapper.run(params)
        fallback.run.assert_called_once_with(params)


# ---------------------------------------------------------------------------
# Unknown PROBE_METHOD value
# ---------------------------------------------------------------------------


class TestUnknownProbeMethod:
    def test_unknown_method_raises_before_fallback(self, wrapper: ProbeMethodWrapperMacro, fallback: Mock) -> None:
        params = _make_params(PROBE_METHOD="laser")
        with pytest.raises(ValueError, match="Unknown PROBE_METHOD"):
            wrapper.run(params)
        fallback.run.assert_not_called()


# ---------------------------------------------------------------------------
# Touch mode – non-mesh commands (Z_TILT_ADJUST, QUAD_GANTRY_LEVEL, etc.)
# ---------------------------------------------------------------------------


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
        assert mode_during_fallback[0] is probe.touch, "Mode should be touch during fallback"
        assert probe.current_mode is original_mode, "Mode should be restored after run"

    def test_touch_restores_mode_on_fallback_exception(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        fallback.run.side_effect = RuntimeError("levelling failed")

        original_mode = probe.current_mode
        with pytest.raises(RuntimeError, match="levelling failed"):
            wrapper.run(params)

        assert probe.current_mode is original_mode, "Mode should be restored even after fallback exception"

    def test_touch_does_not_inject_mesh_params(
        self, wrapper: ProbeMethodWrapperMacro, fallback: Mock
    ) -> None:
        """Non-mesh commands must not get MESH_MIN/MESH_MAX injected."""
        params = _make_params(PROBE_METHOD="touch")
        wrapper.run(params)
        called_params: MacroParams = fallback.run.call_args[0][0]
        assert called_params is params, "Non-mesh command must delegate original params unchanged"


# ---------------------------------------------------------------------------
# Touch mode – BED_MESH_CALIBRATE
# ---------------------------------------------------------------------------


class TestTouchModeMesh:
    def test_method_absent_defaults_to_automatic(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_wrapper.run(params)

        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("METHOD", None) == "automatic"

    def test_method_automatic_is_accepted(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch", METHOD="automatic")
        mesh_wrapper.run(params)
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("METHOD", None) == "automatic"

    def test_method_automatic_case_insensitive(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch", METHOD="AUTOMATIC")
        mesh_wrapper.run(params)
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("METHOD", None) == "automatic"

    def test_explicit_method_other_than_automatic_raises_before_fallback(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch", METHOD="scan")
        with pytest.raises(ValueError, match="METHOD=automatic"):
            mesh_wrapper.run(params)
        mesh_fallback.run.assert_not_called()

    def test_default_mesh_bounds_from_touch_boundaries(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_wrapper.run(params)

        boundaries = probe.touch.boundaries
        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("MESH_MIN", None) == f"{boundaries.min_x},{boundaries.min_y}"
        assert called_params.get("MESH_MAX", None) == f"{boundaries.max_x},{boundaries.max_y}"

    def test_explicit_mesh_min_max_preserved(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock
    ) -> None:
        params = _make_params(PROBE_METHOD="touch", MESH_MIN="10,10", MESH_MAX="190,190")
        mesh_wrapper.run(params)

        called_params: MacroParams = mesh_fallback.run.call_args[0][0]
        assert called_params.get("MESH_MIN", None) == "10,10"
        assert called_params.get("MESH_MAX", None) == "190,190"

    def test_touch_enters_touch_context_during_fallback(
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

    def test_touch_restores_mode_on_fallback_exception(
        self, mesh_wrapper: ProbeMethodWrapperMacro, mesh_fallback: Mock, probe: Probe
    ) -> None:
        params = _make_params(PROBE_METHOD="touch")
        mesh_fallback.run.side_effect = RuntimeError("mesh failed")

        original_mode = probe.current_mode
        with pytest.raises(RuntimeError, match="mesh failed"):
            mesh_wrapper.run(params)

        assert probe.current_mode is original_mode
