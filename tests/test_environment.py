from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestDetectEnvironment:
    def test_detects_kalico(self, mocker: MockerFixture) -> None:
        """When klippy.APP_NAME is 'Kalico', detect Kalico."""
        klippy_mock = MagicMock()
        klippy_mock.APP_NAME = "Kalico"
        mocker.patch.dict(sys.modules, {"klippy": klippy_mock})

        from cartographer.runtime.environment import Environment, detect_environment

        assert detect_environment(None) == Environment.Kalico

    def test_detects_klipper_v12_when_start_probe_session_missing(self, mocker: MockerFixture) -> None:
        """When PrinterProbe lacks start_probe_session, detect KlipperV12."""
        # Stub klippy without APP_NAME (or not Kalico)
        klippy_mock = MagicMock(spec=[])
        mocker.patch.dict(sys.modules, {"klippy": klippy_mock})

        # Stub extras.probe with PrinterProbe that lacks start_probe_session
        probe_mock = MagicMock()
        probe_mock.PrinterProbe = type("PrinterProbe", (), {})
        mocker.patch.dict(sys.modules, {"extras": MagicMock(), "extras.probe": probe_mock})

        from cartographer.runtime.environment import Environment, detect_environment

        assert detect_environment(None) == Environment.KlipperV12

    def test_detects_klipper_when_start_probe_session_present(self, mocker: MockerFixture) -> None:
        """When PrinterProbe has start_probe_session, detect modern Klipper."""
        klippy_mock = MagicMock(spec=[])
        mocker.patch.dict(sys.modules, {"klippy": klippy_mock})

        # Stub extras.probe with PrinterProbe that has start_probe_session
        probe_mock = MagicMock()
        probe_mock.PrinterProbe = type("PrinterProbe", (), {"start_probe_session": None})
        mocker.patch.dict(sys.modules, {"extras": MagicMock(), "extras.probe": probe_mock})

        from cartographer.runtime.environment import Environment, detect_environment

        assert detect_environment(None) == Environment.Klipper
