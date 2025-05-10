from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, TypeVar

import pytest
from typing_extensions import ParamSpec, override

from cartographer.interfaces.multiprocessing import TaskExecutor
from cartographer.interfaces.printer import MacroParams, Mcu, Position, Sample, Toolhead
from cartographer.stream import Session

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

P = ParamSpec("P")
R = TypeVar("R")


collect_ignore: list[str] = []
if sys.version_info < (3, 9):
    # pytest-bdd 8.0.0 requires Python 3.9+
    collect_ignore.append("bdd")


@pytest.fixture
def toolhead(mocker: MockerFixture) -> Toolhead:
    mock = mocker.MagicMock(spec=Toolhead, autospec=True, instance=True)

    def get_position() -> Position:
        return Position(x=10, y=10, z=5)

    def apply_axis_twist_compensation(position: Position) -> Position:
        return position

    mock.get_position = get_position
    mock.apply_axis_twist_compensation = apply_axis_twist_compensation

    return mock


@pytest.fixture
def session(mocker: MockerFixture) -> Session[Sample]:
    return Session(mocker.Mock(), mocker.Mock())


@pytest.fixture
def mcu(mocker: MockerFixture, session: Session[Sample]) -> Mcu[object, Sample]:
    mock = mocker.MagicMock(spec=Mcu, autospec=True, instance=True)
    mock.start_session = mocker.Mock(return_value=session)
    return mock


@pytest.fixture
def params(mocker: MockerFixture) -> MacroParams:
    return mocker.MagicMock(spec=MacroParams, autospec=True, instance=True)


class InlineTaskExecutor(TaskExecutor):
    @override
    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)


@pytest.fixture
def task_executor() -> TaskExecutor:
    return InlineTaskExecutor()
