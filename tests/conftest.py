from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.printer import (
    Endstop,
    HomingState,
    MacroParams,
    Mcu,
    Position,
    Sample,
    TemperatureStatus,
    Toolhead,
)
from cartographer.macros.axis_twist_compensation import AxisTwistCompensationAdapter
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode, ScanModeConfiguration
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration
from cartographer.stream import Session
from tests.mocks.config import MockConfiguration
from tests.mocks.params import MockParams
from tests.mocks.task_executor import InlineTaskExecutor

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import TaskExecutor

collect_ignore: list[str] = []
if sys.version_info < (3, 9):
    # pytest-bdd 8.0.0 requires Python 3.9+
    collect_ignore.append("bdd")


@pytest.fixture
def toolhead(mocker: MockerFixture) -> Toolhead:
    mock = mocker.MagicMock(spec=Toolhead, autospec=True, instance=True)

    def get_position() -> Position:
        return Position(x=10, y=10, z=5)

    def get_extruder_temperature() -> TemperatureStatus:
        return TemperatureStatus(30, 30)

    def z_home_end(endstop: Endstop) -> None:
        homing_state = mocker.Mock(spec=HomingState, autospec=True)
        homing_state.is_homing_z = mocker.Mock(return_value=True)
        endstop.on_home_end(homing_state)

    mock.get_axis_limits = mocker.Mock(return_value=(0, 100))
    mock.get_position = get_position
    mock.get_extruder_temperature = get_extruder_temperature
    mock.z_home_end = z_home_end
    last_move_time = 0

    def get_last_move_time() -> float:
        nonlocal last_move_time
        last_move_time += 1
        return last_move_time

    mock.get_last_move_time = get_last_move_time

    mock.z_probing_move = mocker.Mock(return_value=0)

    return mock


@pytest.fixture
def axis_twist_compensation_adapter(mocker: MockerFixture) -> AxisTwistCompensationAdapter:
    mock = mocker.MagicMock(spec=AxisTwistCompensationAdapter)

    def get_z_compensation_value(x: float, y: float) -> float:
        del x, y
        return 0

    mock.get_z_compensation_value = get_z_compensation_value
    return mock


@pytest.fixture
def session(mocker: MockerFixture) -> Session[Sample]:
    return Session(mocker.Mock(), mocker.Mock())


@pytest.fixture
def mcu(mocker: MockerFixture, session: Session[Sample]) -> Mcu:
    mock = mocker.MagicMock(spec=Mcu, autospec=True, instance=True)
    mock.start_session = mocker.Mock(return_value=session)
    return mock


@pytest.fixture
def params() -> MacroParams:
    return MockParams()


@pytest.fixture
def config() -> Configuration:
    return MockConfiguration()


@pytest.fixture
def scan(
    mcu: Mcu, toolhead: Toolhead, axis_twist_compensation_adapter: AxisTwistCompensationAdapter, config: Configuration
):
    return ScanMode(mcu, toolhead, ScanModeConfiguration.from_config(config), None, axis_twist_compensation_adapter)


@pytest.fixture
def touch(mcu: Mcu, toolhead: Toolhead, config: Configuration):
    return TouchMode(mcu, toolhead, TouchModeConfiguration.from_config(config))


@pytest.fixture
def probe(scan: ScanMode, touch: TouchMode) -> Probe:
    return Probe(scan, touch)


@pytest.fixture
def task_executor() -> TaskExecutor:
    return InlineTaskExecutor()


@pytest.fixture
def homing_state(mocker: MockerFixture) -> HomingState:
    return mocker.Mock(spec=HomingState, autospec=True)
