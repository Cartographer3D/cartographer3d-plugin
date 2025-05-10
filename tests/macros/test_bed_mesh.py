from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypeAlias

from cartographer.interfaces.printer import Position, Toolhead
from cartographer.macros.bed_mesh import BedMeshCalibrateMacro, Configuration, MeshHelper, MeshPoint
from cartographer.probe import Probe, ScanMode
from cartographer.probe.scan_mode import Model
from cartographer.probe.touch_mode import TouchMode

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.interfaces.multiprocessing import TaskExecutor
    from cartographer.stream import Session


@dataclass
class Sample:
    time: float
    frequency: float
    position: Position | None = field(default_factory=lambda: Position(10, 10, 10))
    velocity: float | None = None


Macro: TypeAlias = BedMeshCalibrateMacro
Helper: TypeAlias = MeshHelper


@pytest.fixture
def scan_model(mocker: MockerFixture) -> Model:
    return mocker.MagicMock(spec=Model, autospec=True, instance=True)


@pytest.fixture
def offset() -> Position:
    return Position(0, 0, 0)


@pytest.fixture
def scan_mode(mocker: MockerFixture, scan_model: Model, session: Session[Sample], offset: Position) -> ScanMode:
    scan_mode = mocker.MagicMock(spec=ScanMode, autospec=True, instance=True)
    scan_mode.model = scan_model
    scan_mode.offset = offset
    scan_mode.probe_height = 10.0
    scan_mode.start_session = mocker.Mock(return_value=session)
    return scan_mode


@pytest.fixture
def touch_mode(mocker: MockerFixture) -> TouchMode:
    touch_mode = mocker.MagicMock(spec=TouchMode, autospec=True, instance=True)
    return touch_mode


@pytest.fixture
def probe(scan_mode: ScanMode, touch_mode: TouchMode) -> Probe:
    return Probe(scan_mode, touch_mode)


@pytest.fixture
def helper(mocker: MockerFixture) -> Helper:
    return mocker.MagicMock(spec=MeshHelper, autospec=True, instance=True)


class MockConfiguration(Configuration):
    scan_speed: float = 400
    scan_mesh_runs: int = 1
    scan_height: float = 5.0


@pytest.fixture
def config() -> Configuration:
    return MockConfiguration()


@pytest.fixture
def macro(
    probe: Probe,
    toolhead: Toolhead,
    helper: Helper,
    task_executor: TaskExecutor,
    config: Configuration,
) -> Macro:
    return BedMeshCalibrateMacro(probe, toolhead, helper, task_executor, config)


def test_run_valid_scan(mocker: MockerFixture, macro: Macro, toolhead: Toolhead, helper: Helper):
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, False), MeshPoint(20, 20, False)])
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    params.get_float = mocker.Mock(return_value=42.0)
    move_spy = mocker.spy(toolhead, "move")
    finalize_spy = mocker.spy(helper, "finalize")

    macro.run(params)

    assert [
        mocker.call(x=10, y=10, speed=42.0),
        mocker.call(x=20, y=20, speed=42.0),
    ] in move_spy.mock_calls
    assert finalize_spy.call_count == 1


def test_applies_offsets(
    mocker: MockerFixture,
    macro: Macro,
    toolhead: Toolhead,
    helper: Helper,
    offset: Position,
):
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, False), MeshPoint(20, 20, False)])
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    params.get_float = mocker.Mock(return_value=42.0)
    offset.x = -5
    offset.y = 5
    move_spy = mocker.spy(toolhead, "move")

    macro.run(params)

    assert [
        mocker.call(x=15, y=5, speed=42.0),
        mocker.call(x=25, y=15, speed=42.0),
    ] in move_spy.mock_calls


def test_multiple_runs(
    mocker: MockerFixture,
    macro: Macro,
    toolhead: Toolhead,
    helper: Helper,
):
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, False), MeshPoint(20, 20, False)])
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    params.get_int = mocker.Mock(return_value=3)
    params.get_float = mocker.Mock(return_value=42.0)
    move_spy = mocker.spy(toolhead, "move")

    macro.run(params)

    assert [
        # first
        mocker.call(x=10, y=10, speed=42.0),
        mocker.call(x=20, y=20, speed=42.0),
        # second
        mocker.call(x=20, y=20, speed=42.0),
        mocker.call(x=10, y=10, speed=42.0),
        # third
        mocker.call(x=10, y=10, speed=42.0),
        mocker.call(x=20, y=20, speed=42.0),
    ] in move_spy.mock_calls


def test_run_invalid_method(mocker: MockerFixture, macro: Macro, helper: Helper):
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="invalid")
    orig_macro_spy = mocker.spy(helper, "orig_macro")
    finalize_spy = mocker.spy(helper, "finalize")

    macro.run(params)
    assert orig_macro_spy.call_count == 1
    assert finalize_spy.call_count == 0


def test_calculate_positions_no_valid_samples(
    mocker: MockerFixture,
    macro: Macro,
    scan_model: Model,
    helper: Helper,
    session: Session[Sample],
):
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, True)])
    scan_model.frequency_to_distance = mocker.Mock(return_value=float("nan"))
    session.get_items = lambda: [Sample(time=0.0, frequency=100) for _ in range(2)]

    with pytest.raises(RuntimeError, match="no valid samples"):
        macro.run(params)


def test_calculate_positions_cluster_no_samples(
    mocker: MockerFixture,
    macro: Macro,
    scan_model: Model,
    helper: Helper,
    session: Session[Sample],
):
    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, True), MeshPoint(20, 20, True)])
    scan_model.frequency_to_distance = mocker.Mock(return_value=1)
    session.get_items = lambda: [Sample(time=0.0, frequency=100) for _ in range(2)]

    with pytest.raises(RuntimeError, match="no samples"):
        macro.run(params)


def test_finalize_with_positions(
    mocker: MockerFixture,
    macro: Macro,
    scan_mode: ScanMode,
    scan_model: Model,
    helper: Helper,
    session: Session[Sample],
):
    scan_height = 5

    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    params.get_float = mocker.Mock(return_value=scan_height)
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, True), MeshPoint(20, 20, True)])
    session.get_items = lambda: [
        Sample(time=0.0, frequency=1, position=Position(10 + 10 * i, 10 + 10 * i, 5)) for i in range(2)
    ]

    distances = [1, 2]
    positions = [Position(10, 10, scan_height), Position(20, 20, scan_height)]
    scan_model.frequency_to_distance = mocker.Mock(side_effect=distances)

    finalize_spy = mocker.spy(helper, "finalize")
    expected_positions = [
        Position(pos.x, pos.y, scan_height + scan_mode.probe_height - dist) for pos, dist in zip(positions, distances)
    ]

    macro.run(params)

    assert finalize_spy.mock_calls == [mocker.call(Position(0, 0, 0), expected_positions)]


def test_probe_applies_axis_twist_compensation(
    mocker: MockerFixture,
    macro: Macro,
    scan_model: Model,
    toolhead: Toolhead,
    helper: Helper,
    session: Session[Sample],
):
    z_comp = 0.5
    distances = [1, 2]
    positions = [Position(10, 10, 5), Position(20, 20, 5)]

    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, True), MeshPoint(20, 20, True)])
    session.get_items = lambda: [
        Sample(time=0.0, frequency=1, position=Position(10 + 10 * i, 10 + 10 * i, 5)) for i in range(2)
    ]
    scan_model.frequency_to_distance = mocker.Mock(side_effect=distances)
    toolhead.apply_axis_twist_compensation = lambda position: Position(position.x, position.y, z_comp)

    finalize_spy = mocker.spy(helper, "finalize")
    expected_positions = [Position(pos.x, pos.y, z_comp) for pos in positions]

    macro.run(params)

    assert finalize_spy.mock_calls == [mocker.call(Position(0, 0, 0), expected_positions)]


def test_ignores_unincluded_points(
    mocker: MockerFixture,
    scan_mode: ScanMode,
    macro: Macro,
    scan_model: Model,
    helper: Helper,
    session: Session[Sample],
):
    distances = [1, 1]
    scan_height = 5
    dist = 1

    params = mocker.MagicMock()
    params.get = mocker.Mock(return_value="scan")
    params.get_float = mocker.Mock(return_value=scan_height)
    helper.prepare_scan_path = mocker.Mock(return_value=[MeshPoint(10, 10, False), MeshPoint(10.3, 10.3, True)])
    session.get_items = lambda: [Sample(time=0.0, frequency=dist) for _ in range(2)]
    scan_model.frequency_to_distance = mocker.Mock(side_effect=distances)

    finalize_spy = mocker.spy(helper, "finalize")

    macro.run(params)

    assert finalize_spy.mock_calls == [
        mocker.call(Position(0, 0, 0), [Position(10.3, 10.3, scan_height + scan_mode.probe_height - dist)])
    ]
