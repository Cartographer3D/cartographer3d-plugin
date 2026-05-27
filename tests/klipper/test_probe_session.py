from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.adapters.klipper.probe import KlipperProbeSession
from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def probe_mode(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.perform_probe = mocker.Mock(return_value=1.5)
    mock.offset = Position(0, 0, 0)
    mock.note_homing_complete = mocker.Mock()
    return mock


@pytest.fixture
def toolhead(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.get_position = mocker.Mock(return_value=Position(10, 20, 5))
    return mock


@pytest.fixture
def session(probe_mode: Mock, toolhead: Mock) -> KlipperProbeSession:
    return KlipperProbeSession(probe_mode, toolhead)


class TestRunProbe:
    def test_calls_perform_probe(self, session: KlipperProbeSession, probe_mode: Mock) -> None:
        gcmd = Mock()
        gcmd.get = Mock(return_value=None)

        session.run_probe(gcmd)

        probe_mode.perform_probe.assert_called_once()

    def test_stores_result(self, session: KlipperProbeSession) -> None:
        gcmd = Mock()
        gcmd.get = Mock(return_value=None)

        session.run_probe(gcmd)

        results = session.pull_probed_results()
        assert len(results) == 1
        assert results[0] == [10, 20, 1.5]

    def test_calls_note_homing_complete_when_home_attempt_num_present(
        self, session: KlipperProbeSession, probe_mode: Mock
    ) -> None:
        gcmd = Mock()

        def get_param(name: str, default: str | None = None) -> str | None:
            return "0" if name == "HOME_ATTEMPT_NUM" else default

        gcmd.get = Mock(side_effect=get_param)

        session.run_probe(gcmd)

        probe_mode.note_homing_complete.assert_called_once()

    def test_does_not_call_note_homing_complete_for_regular_probing(
        self, session: KlipperProbeSession, probe_mode: Mock
    ) -> None:
        gcmd = Mock()
        gcmd.get = Mock(return_value=None)

        session.run_probe(gcmd)

        probe_mode.note_homing_complete.assert_not_called()


class TestEndProbeSession:
    def test_clears_results(self, session: KlipperProbeSession) -> None:
        gcmd = Mock()
        gcmd.get = Mock(return_value=None)

        session.run_probe(gcmd)
        session.end_probe_session()

        assert session.pull_probed_results() == []
