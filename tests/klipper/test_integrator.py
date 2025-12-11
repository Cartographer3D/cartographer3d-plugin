from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from cartographer.adapters.klipper.endstop import KlipperEndstop, KlipperProbeEndstop
from cartographer.adapters.klipper.homing import KlipperHomingChip
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def probe_class(mocker: MockerFixture) -> Mock:
    return mocker.Mock()


@pytest.fixture
def adapters(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.mcu = mocker.Mock()
    mock.printer = mocker.Mock()
    mock.config = mocker.Mock()
    mock.toolhead = mocker.Mock()
    mock.printer.lookup_object = mocker.Mock(return_value=mocker.Mock())
    return mock


@pytest.fixture
def integrator(adapters: Mock, probe_class: Mock) -> KlipperLikeIntegrator:
    return KlipperLikeIntegrator(adapters, probe_class)


class TestRegisterEndstopPin:
    def test_probe_chip_registers_probe_endstop(self, integrator: KlipperLikeIntegrator, adapters: Mock) -> None:
        endstop = Mock()
        pins_mock = adapters.printer.lookup_object.return_value

        integrator.register_endstop_pin("probe", "z_virtual_endstop", endstop)

        pins_mock.register_chip.assert_called_once()
        chip = pins_mock.register_chip.call_args[0][1]
        assert isinstance(chip, KlipperHomingChip)
        assert isinstance(chip.endstop, KlipperProbeEndstop)
        assert hasattr(chip.endstop, "get_position_endstop")

    def test_non_probe_chip_registers_plain_endstop(self, integrator: KlipperLikeIntegrator, adapters: Mock) -> None:
        endstop = Mock()
        pins_mock = adapters.printer.lookup_object.return_value

        integrator.register_endstop_pin("cartographer_probe", "z_virtual_endstop", endstop)

        pins_mock.register_chip.assert_called_once()
        chip = pins_mock.register_chip.call_args[0][1]
        assert isinstance(chip, KlipperHomingChip)
        assert isinstance(chip.endstop, KlipperEndstop)
        assert not hasattr(chip.endstop, "get_position_endstop")


class TestRegisterProbe:
    def test_invokes_probe_callable_with_expected_deps(
        self, integrator: KlipperLikeIntegrator, adapters: Mock, probe_class: Mock
    ) -> None:
        cartographer = Mock()

        integrator.register_probe(cartographer)

        probe_class.assert_called_once_with(
            adapters.toolhead,
            cartographer.probe,
            cartographer.probe_macro,
            cartographer.query_probe_macro,
            cartographer.config.general,
        )

    def test_registers_probe_instance_as_printer_object(
        self, integrator: KlipperLikeIntegrator, adapters: Mock, probe_class: Mock
    ) -> None:
        cartographer = Mock()

        integrator.register_probe(cartographer)

        adapters.printer.add_object.assert_called_once_with("probe", probe_class.return_value)
