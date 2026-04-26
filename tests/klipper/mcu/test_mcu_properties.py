from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

# Stub klipper modules not present in test environment
for _name in ("mcu", "reactor", "configfile", "motion_report", "extras", "extras.thermistor", "greenlet"):
    if _name not in sys.modules:
        sys.modules[_name] = Mock()

from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu  # noqa: E402

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_mcu(klipper_mcu: Mock, commands: object | None, constants: object | None) -> KlipperCartographerMcu:
    """Build a bare KlipperCartographerMcu without running __init__."""
    obj = object.__new__(KlipperCartographerMcu)
    obj.klipper_mcu = klipper_mcu
    obj._commands = commands  # pyright: ignore [reportAttributeAccessIssue, reportPrivateUsage]
    obj._constants = constants  # pyright: ignore [reportAttributeAccessIssue, reportPrivateUsage]
    return obj


class TestCommandsProperty:
    def test_returns_commands_when_connected(self, mocker: MockerFixture) -> None:
        klipper_mcu = mocker.MagicMock()
        klipper_mcu.non_critical_disconnected = False
        commands = mocker.Mock()
        obj = _make_mcu(klipper_mcu, commands=commands, constants=None)

        assert obj.commands is commands

    def test_raises_when_not_initialized(self, mocker: MockerFixture) -> None:
        klipper_mcu = mocker.MagicMock()
        obj = _make_mcu(klipper_mcu, commands=None, constants=None)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = obj.commands

    def test_raises_when_disconnected(self, mocker: MockerFixture) -> None:
        klipper_mcu = mocker.MagicMock()
        klipper_mcu.non_critical_disconnected = True
        commands = mocker.Mock()
        obj = _make_mcu(klipper_mcu, commands=commands, constants=None)

        with pytest.raises(RuntimeError, match="disconnected"):
            _ = obj.commands

    def test_works_on_stock_klipper_without_attribute(self, mocker: MockerFixture) -> None:
        """Stock Klipper has no non_critical_disconnected attribute."""
        klipper_mcu = mocker.MagicMock(spec=[])
        commands = mocker.Mock()
        obj = _make_mcu(klipper_mcu, commands=commands, constants=None)

        assert obj.commands is commands


class TestConstantsProperty:
    def test_returns_constants_when_initialized(self, mocker: MockerFixture) -> None:
        constants = mocker.Mock()
        obj = _make_mcu(mocker.MagicMock(), commands=None, constants=constants)

        assert obj.constants is constants

    def test_raises_when_not_initialized(self, mocker: MockerFixture) -> None:
        obj = _make_mcu(mocker.MagicMock(), commands=None, constants=None)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = obj.constants
