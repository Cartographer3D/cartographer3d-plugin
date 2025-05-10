from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typing_extensions import override

from cartographer.adapters.shared.endstop import KlipperEndstop
from cartographer.interfaces.printer import Endstop, HomingState

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.adapters.shared.mcu import KlipperCartographerMcu

ReactorCompletion = MagicMock


class MockEndstop(Endstop[ReactorCompletion]):
    @override
    def query_is_triggered(self, print_time: float) -> bool:
        return False

    @override
    def home_start(self, print_time: float) -> ReactorCompletion:
        return MagicMock()

    @override
    def home_wait(self, home_end_time: float) -> float:
        return home_end_time

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        return

    @override
    def get_endstop_position(self) -> float:
        return 2.0


@pytest.fixture
def mcu(mocker: MockerFixture) -> KlipperCartographerMcu:
    return mocker.MagicMock()


@pytest.fixture
def endstop() -> Endstop[object]:
    return MockEndstop()


def test_endstop_is_memoized(mcu: KlipperCartographerMcu, endstop: Endstop[ReactorCompletion]):
    a = KlipperEndstop(mcu, endstop)
    b = KlipperEndstop(mcu, endstop)

    assert a == b


def test_endstop_creates_multiple(mcu: KlipperCartographerMcu, endstop: Endstop[ReactorCompletion]):
    another_endstop = MockEndstop()
    a = KlipperEndstop(mcu, endstop)
    b = KlipperEndstop(mcu, another_endstop)

    assert a != b
