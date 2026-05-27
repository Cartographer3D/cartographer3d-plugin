from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from typing_extensions import override

from cartographer.adapters.klipper.endstop import KlipperEndstop, KlipperProbeEndstop
from cartographer.adapters.klipper.homing import KlipperHomingChip
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class ConcreteIntegrator(KlipperLikeIntegrator):
    """Minimal concrete subclass for testing the ABC's methods."""

    @override
    def register_probe(self, cartographer: object) -> None:
        pass


@pytest.fixture
def adapters(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.mcu = mocker.Mock()
    mock.printer = mocker.Mock()
    mock.config = mocker.Mock()
    mock.printer.lookup_object = mocker.Mock(return_value=mocker.Mock())
    return mock


@pytest.fixture
def integrator(adapters: Mock) -> ConcreteIntegrator:
    return ConcreteIntegrator(adapters)


class TestRegisterEndstopPin:
    def test_probe_chip_registers_probe_endstop(self, integrator: ConcreteIntegrator, adapters: Mock) -> None:
        endstop = Mock()
        pins_mock = adapters.printer.lookup_object.return_value

        integrator.register_endstop_pin("probe", "z_virtual_endstop", endstop)

        pins_mock.register_chip.assert_called_once()
        chip = pins_mock.register_chip.call_args[0][1]
        assert isinstance(chip, KlipperHomingChip)
        assert isinstance(chip.endstop, KlipperProbeEndstop)
        assert hasattr(chip.endstop, "get_position_endstop")

    def test_non_probe_chip_registers_plain_endstop(self, integrator: ConcreteIntegrator, adapters: Mock) -> None:
        endstop = Mock()
        pins_mock = adapters.printer.lookup_object.return_value

        integrator.register_endstop_pin("cartographer_probe", "z_virtual_endstop", endstop)

        pins_mock.register_chip.assert_called_once()
        chip = pins_mock.register_chip.call_args[0][1]
        assert isinstance(chip, KlipperHomingChip)
        assert isinstance(chip.endstop, KlipperEndstop)
        assert not hasattr(chip.endstop, "get_position_endstop")
