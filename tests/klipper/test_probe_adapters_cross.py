"""Cross-adapter test: verify all probe classes accept Probe (not ProbeMode) and
use current_mode consistently for offsets and probing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper.probe import KlipperCartographerProbe
from cartographer.adapters.klipper_v12.probe import KlipperV12CartographerProbe
from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def probe_mode(mocker: MockerFixture) -> Mock:
    """A mock ProbeMode with a known offset and trigger position."""
    mode = mocker.Mock()
    mode.offset = Position(0.1, 0.2, -0.5)
    mode.perform_probe = mocker.Mock(return_value=2.5)
    mode.note_homing_complete = mocker.Mock()
    return mode


@pytest.fixture
def probe(mocker: MockerFixture, probe_mode: Mock) -> Mock:
    """A mock Probe whose current_mode is the probe_mode fixture."""
    p = mocker.Mock()
    p.current_mode = probe_mode
    p.perform_probe = probe_mode.perform_probe
    return p


@pytest.fixture
def toolhead(mocker: MockerFixture) -> Mock:
    th = mocker.Mock()
    th.get_position = mocker.Mock(return_value=Position(1.0, 2.0, 5.0))
    return th


@pytest.fixture
def config(mocker: MockerFixture) -> Mock:
    cfg = mocker.Mock()
    cfg.lift_speed = 5.0
    return cfg


@pytest.fixture
def probe_macro(mocker: MockerFixture) -> Mock:
    m = mocker.Mock()
    m.last_triggered = False
    m.last_trigger_position = 0.0
    m.last_probe_position = None
    return m


@pytest.fixture
def query_probe_macro(mocker: MockerFixture) -> Mock:
    m = mocker.Mock()
    m.last_triggered = False
    return m


class TestKlipperCartographerProbe:
    def test_get_offsets_delegates_to_current_mode(
        self,
        probe: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        p = KlipperCartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        offsets = p.get_offsets()
        assert offsets == (0.1, 0.2, -0.5)

    def test_start_probe_session_uses_current_mode(
        self,
        probe: Mock,
        probe_mode: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        from cartographer.adapters.klipper.probe import KlipperProbeSession

        p = KlipperCartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        gcmd = Mock()
        session = p.start_probe_session(gcmd)
        assert isinstance(session, KlipperProbeSession)
        # Verify the session delegates to probe.current_mode by running a probe
        gcmd.get = Mock(return_value=None)
        session.run_probe(gcmd)
        probe_mode.perform_probe.assert_called_once()


class TestKalicoCartographerProbe:
    def test_get_offsets_delegates_to_current_mode(
        self,
        probe: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        p = KalicoCartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        offsets = p.get_offsets()
        assert offsets == (0.1, 0.2, -0.5)

    def test_run_probe_uses_current_mode(
        self,
        probe: Mock,
        probe_mode: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        p = KalicoCartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        gcmd = Mock()
        result = p.run_probe(gcmd)
        probe_mode.perform_probe.assert_called_once()
        assert result == [1.0, 2.0, 2.5]


class TestKlipperV12CartographerProbe:
    def test_get_offsets_delegates_to_current_mode(
        self,
        probe: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        p = KlipperV12CartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        offsets = p.get_offsets()
        assert offsets == (0.1, 0.2, -0.5)

    def test_run_probe_uses_current_mode(
        self,
        probe: Mock,
        probe_mode: Mock,
        toolhead: Mock,
        probe_macro: Mock,
        query_probe_macro: Mock,
        config: Mock,
    ) -> None:
        p = KlipperV12CartographerProbe(toolhead, probe, probe_macro, query_probe_macro, config)
        gcmd = Mock()
        result = p.run_probe(gcmd)
        probe_mode.perform_probe.assert_called_once()
        assert result == [1.0, 2.0, 2.5]
