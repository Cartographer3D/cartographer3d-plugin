from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.adapters.klipper.endstop import KlipperEndstop, KlipperEndstopBase, KlipperProbeEndstop

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


@pytest.fixture
def mcu(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.host_mcu = mocker.Mock()
    mock.dispatch = mocker.Mock()
    return mock


@pytest.fixture
def endstop(mocker: MockerFixture) -> Mock:
    mock = mocker.Mock()
    mock.get_endstop_position = mocker.Mock(return_value=2.0)
    return mock


class TestKlipperProbeEndstop:
    def test_has_get_position_endstop(self, mcu: Mock, endstop: Mock) -> None:
        es = KlipperProbeEndstop(mcu, endstop)
        assert hasattr(es, "get_position_endstop")

    def test_get_position_endstop_returns_value(self, mcu: Mock, endstop: Mock) -> None:
        es = KlipperProbeEndstop(mcu, endstop)
        assert es.get_position_endstop() == 2.0

    def test_is_instance_of_base(self, mcu: Mock, endstop: Mock) -> None:
        es = KlipperProbeEndstop(mcu, endstop)
        assert isinstance(es, KlipperEndstopBase)


class TestKlipperEndstop:
    def test_does_not_have_get_position_endstop(self, mcu: Mock, endstop: Mock) -> None:
        es = KlipperEndstop(mcu, endstop)
        assert not hasattr(es, "get_position_endstop")

    def test_is_instance_of_base(self, mcu: Mock, endstop: Mock) -> None:
        es = KlipperEndstop(mcu, endstop)
        assert isinstance(es, KlipperEndstopBase)
