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

    def test_detects_klipper_v12_when_trigger_dispatch_missing(self, mocker: MockerFixture) -> None:
        """When mcu module exists but lacks TriggerDispatch, detect KlipperV12."""
        # Stub klippy without APP_NAME (or not Kalico)
        klippy_mock = MagicMock(spec=[])
        mocker.patch.dict(sys.modules, {"klippy": klippy_mock})

        # Stub mcu module without TriggerDispatch
        mcu_mock = MagicMock(spec=["MCU", "MCU_trsync", "get_printer_mcu"])
        mocker.patch.dict(sys.modules, {"mcu": mcu_mock})

        from cartographer.runtime.environment import Environment, detect_environment

        assert detect_environment(None) == Environment.KlipperV12

    def test_detects_klipper_when_trigger_dispatch_present(self, mocker: MockerFixture) -> None:
        """When mcu.TriggerDispatch exists, detect modern Klipper."""
        klippy_mock = MagicMock(spec=[])
        mocker.patch.dict(sys.modules, {"klippy": klippy_mock})

        # Stub mcu module WITH TriggerDispatch
        mcu_mock = MagicMock()
        mcu_mock.TriggerDispatch = MagicMock()
        mocker.patch.dict(sys.modules, {"mcu": mcu_mock})

        from cartographer.runtime.environment import Environment, detect_environment

        assert detect_environment(None) == Environment.Klipper
