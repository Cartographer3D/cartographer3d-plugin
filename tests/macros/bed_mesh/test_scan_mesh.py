from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final

import numpy as np
import pytest
from typing_extensions import override

from cartographer.interfaces.printer import Position, Sample, Toolhead
from cartographer.macros.bed_mesh.interfaces import BedMeshAdapter
from cartographer.macros.bed_mesh.scan_mesh import BedMeshCalibrateConfiguration, BedMeshCalibrateMacro
from tests.mocks.config import MockConfiguration, default_general_config
from tests.mocks.task_executor import InlineTaskExecutor
from tests.mocks.toolhead import MockToolhead

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe.probe import Probe
    from cartographer.stream import Session
    from tests.mocks.params import MockParams


@final
class MockProbe:
    """Mock probe with scan offset."""

    @dataclass
    class MockScan:
        offset: Position
        session: Session[Sample]

        def calculate_sample_distance(self, sample: Sample) -> float:
            """Mock distance calculation - just return z position."""
            return sample.position.z if sample.position else 0.0

        def start_session(self) -> Session[Sample]:
            return self.session

    def __init__(self, session: Session[Sample], offset: Position):
        self.scan = self.MockScan(offset, session)


class MockBedMeshAdapter(BedMeshAdapter):
    """Mock bed mesh adapter."""

    def __init__(self):
        self.mesh_positions: list[Position] = []
        self.mesh_profile: str = ""
        self.objects: list[list[tuple[float, float]]] = []

    @override
    def clear_mesh(self):
        self.mesh_positions = []

    @override
    def apply_mesh(self, mesh_points: list[Position], profile_name: str | None = None):
        self.mesh_positions = mesh_points.copy()
        self.mesh_profile = profile_name or ""

        coords = np.asarray([p.as_tuple() for p in mesh_points])

        xs_rounded = np.round(coords[:, 0], 2)
        ys_rounded = np.round(coords[:, 1], 2)
        zs = coords[:, 2]

        x_unique = np.unique(xs_rounded)
        y_unique = np.unique(ys_rounded)
        points_per_x = len(x_unique)
        points_per_y = len(y_unique)

        x_indices = {v: i for i, v in enumerate(x_unique)}
        y_indices = {v: i for i, v in enumerate(y_unique)}

        matrix = np.full((points_per_y, points_per_x), np.nan)

        for x, y, z in zip(xs_rounded, ys_rounded, zs):
            xi = x_indices[x]
            yi = y_indices[y]
            matrix[yi, xi] = z

        if np.isnan(matrix).any():
            msg = "Mesh has missing points or inconsistent coordinates"
            raise RuntimeError(msg)

    @override
    def get_objects(self) -> list[list[tuple[float, float]]]:
        """Return object points for adaptive meshing."""
        return self.objects

    def set_objects(self, objects: list[list[tuple[float, float]]]):
        """Helper to set objects for testing."""
        self.objects = objects


class TestBedMeshIntegration:
    """Integration tests for bed mesh calibration."""

    @pytest.fixture
    def probe_offset(self):
        """Standard probe offset for testing."""
        return Position(x=5.0, y=3.0, z=0.0)

    @pytest.fixture
    def mesh_config(self):
        """Standard bed mesh configuration."""
        return BedMeshCalibrateConfiguration(
            mesh_min=(10.0, 10.0),
            mesh_max=(90.0, 90.0),
            probe_count=(5, 5),
            speed=100.0,
            adaptive_margin=5.0,
            zero_reference_position=(50.0, 50.0),
            runs=1,
            direction="x",
            height=2.0,
            path="snake",
            faulty_regions=[],
        )

    def create_samples_at_probe_positions(
        self, probe_positions: list[tuple[float, float]], heights: list[float], probe_offset: Position
    ) -> list[Sample]:
        """Create samples that simulate nozzle positions when probe was at specified positions."""
        samples: list[Sample] = []

        for i, (probe_pos, height) in enumerate(zip(probe_positions, heights)):
            # Calculate where nozzle would be when probe is at probe_pos
            nozzle_x = probe_pos[0] - probe_offset.x
            nozzle_y = probe_pos[1] - probe_offset.y

            sample = Sample(
                raw_count=1000,
                time=float(i * 0.1),
                frequency=1000.0,
                temperature=25.0,
                position=Position(x=nozzle_x, y=nozzle_y, z=height),
            )
            samples.append(sample)

        return samples

    @pytest.fixture
    def probe(self, session: Session[Sample], probe_offset: Position):
        return MockProbe(session, probe_offset)

    @pytest.fixture
    def toolhead(self):
        return MockToolhead()

    @pytest.fixture
    def adapter(self):
        return MockBedMeshAdapter()

    @pytest.fixture
    def config(self, probe_offset: Position) -> Configuration:
        return MockConfiguration(
            general=replace(default_general_config, x_offset=probe_offset.x, y_offset=probe_offset.y)
        )

    @pytest.fixture
    def bed_mesh_macro(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: BedMeshAdapter,
        mesh_config: BedMeshCalibrateConfiguration,
    ):
        """Create a bed mesh macro with mocked dependencies."""
        task_executor = InlineTaskExecutor()
        macro = BedMeshCalibrateMacro(probe, toolhead, adapter, None, task_executor, mesh_config)

        return macro

    def test_regular_mesh_boundary_and_coordinate_transformation(
        self,
        mocker: MockerFixture,
        bed_mesh_macro: BedMeshCalibrateMacro,
        probe_offset: Position,
        params: MockParams,
        session: Mock,
        mesh_config: BedMeshCalibrateConfiguration,
        adapter: MockBedMeshAdapter,
        toolhead: MockToolhead,
    ):
        """Test that regular mesh generates correct points and transforms coordinates properly."""

        # Define expected probe positions (5x5 grid from 10,10 to 90,90)
        expected_probe_positions = [(float(10 + 20 * x), float(10 + 20 * y)) for x in range(5) for y in range(5)]

        # Create mock heights for each position
        heights = [1.5 + 0.1 * i for i in range(len(expected_probe_positions))]

        # Create samples that would be generated when probe visits these positions
        mock_samples = self.create_samples_at_probe_positions(expected_probe_positions, heights, probe_offset)
        # Mock the probe session to return our samples
        session.get_items = mocker.Mock(return_value=mock_samples)

        params.params = {"METHOD": "scan"}
        bed_mesh_macro.run(params)

        # Verify that mesh was applied
        assert len(adapter.mesh_positions) == len(expected_probe_positions)

        zero_reference_position = (
            float(mesh_config.zero_reference_position[0]),
            float(mesh_config.zero_reference_position[1]),
        )
        zero_reference_height = mesh_config.height - heights[expected_probe_positions.index(zero_reference_position)]

        # Collect expected mesh positions (in probe coordinates)
        expected_positions = {
            Position(x=px, y=py, z=round(mesh_config.height - z - zero_reference_height, 2))
            for (px, py), z in zip(expected_probe_positions, heights)
        }

        # Collect actual mesh positions (round to avoid float precision errors)
        actual_positions = {Position(x=round(p.x, 2), y=round(p.y, 2), z=round(p.z, 2)) for p in adapter.mesh_positions}

        assert actual_positions == expected_positions

        # Convert toolhead moves to set of (rounded) XY positions
        actual_move_positions = {(round(x, 2), round(y, 2)) for x, y in toolhead.moves}

        # Convert expected probe positions to expected nozzle positions
        expected_nozzle_positions = {
            (round(px - probe_offset.x, 2), round(py - probe_offset.y, 2)) for (px, py) in expected_probe_positions
        }

        # Check that all expected nozzle positions were visited
        missing = expected_nozzle_positions - actual_move_positions
        assert not missing, f"Missing expected nozzle moves: {missing}"

    def test_adaptive_mesh_boundary_and_coordinate_transformation(
        self,
        mocker: MockerFixture,
        bed_mesh_macro: BedMeshCalibrateMacro,
        probe_offset: Position,
        params: MockParams,
        session: Mock,
        mesh_config: BedMeshCalibrateConfiguration,
        adapter: MockBedMeshAdapter,
        toolhead: MockToolhead,
    ):
        """Test adaptive mesh with object boundaries."""

        # Set up objects that should constrain the mesh
        objects = [
            [(25.0, 25.0), (35.0, 25.0), (35.0, 35.0), (25.0, 35.0)],  # Small square
            [(60.0, 60.0), (70.0, 60.0), (70.0, 70.0), (60.0, 70.0)],  # Another square
        ]
        adapter.set_objects(objects)

        # Expected adaptive bounds: object bounds (25,25) to (70,70) + 5 margin = (20,20) to (75,75)
        # With 5x5 base resolution, this should create a 4x4 grid in the adaptive area
        expected_adaptive_positions = [
            (float(20 + 18.33 * x), float(20 + 18.33 * y)) for x in range(4) for y in range(4)
        ]

        # Create mock samples for adaptive positions
        heights = [1.5 + 0.1 * i for i in range(len(expected_adaptive_positions))]
        mock_samples = self.create_samples_at_probe_positions(expected_adaptive_positions, heights, probe_offset)

        # Mock the probe session
        session.get_items = mocker.Mock(return_value=mock_samples)

        params.params = {"METHOD": "scan", "ADAPTIVE": "1"}
        bed_mesh_macro.run(params)

        # Verify adaptive mesh was applied
        assert len(adapter.mesh_positions) == len(expected_adaptive_positions)

        zero_reference_height = -0.32

        # Collect expected mesh positions (in probe coordinates)
        expected_positions = {
            Position(x=round(px, 1), y=round(py, 1), z=round(mesh_config.height - z - zero_reference_height, 2))
            for (px, py), z in zip(expected_adaptive_positions, heights)
        }

        # Collect actual mesh positions (round to avoid float precision errors)
        actual_positions = {Position(x=round(p.x, 1), y=round(p.y, 1), z=round(p.z, 2)) for p in adapter.mesh_positions}

        assert actual_positions == expected_positions

        # Verify that no profile was set for adaptive mesh
        assert adapter.mesh_profile == ""

        # Convert toolhead moves to set of (rounded) XY positions
        actual_move_positions = {(round(x, 1), round(y, 1)) for x, y in toolhead.moves}

        # Convert expected probe positions to expected nozzle positions
        expected_nozzle_positions = {
            (round(px - probe_offset.x, 1), round(py - probe_offset.y, 1)) for (px, py) in expected_adaptive_positions
        }

        # Check that all expected nozzle positions were visited
        missing = expected_nozzle_positions - actual_move_positions
        assert not missing, f"Missing expected nozzle moves: {missing}"

    def test_mesh_with_decimals(
        self,
        mocker: MockerFixture,
        bed_mesh_macro: BedMeshCalibrateMacro,
        probe_offset: Position,
        params: MockParams,
        session: Mock,
        mesh_config: BedMeshCalibrateConfiguration,
        adapter: MockBedMeshAdapter,
        toolhead: MockToolhead,
    ):
        """Test adaptive mesh with object 3 decimal polygons."""

        # Set up objects that should constrain the mesh
        objects = [
            [(25.0, 25.28), (35.0, 25.28), (35.0, 35.0), (25.0, 35.0)],  # Small square
            [(60.0, 60.0), (70.0, 60.0), (70.0, 70.28), (60.0, 70.0)],  # Another square
        ]
        adapter.set_objects(objects)

        # Expected adaptive bounds: object bounds (25,25) to (70,70) + 5 margin = (20,20) to (75,75)
        # With 5x5 base resolution, this should create a 4x4 grid in the adaptive area
        expected_adaptive_positions = [
            (float(20 + 18.33 * x), float(20.28 + 18.33 * y)) for x in range(4) for y in range(4)
        ]

        # Create mock samples for adaptive positions
        heights = [1.5 + 0.1 * i for i in range(len(expected_adaptive_positions))]
        mock_samples = self.create_samples_at_probe_positions(expected_adaptive_positions, heights, probe_offset)

        # Mock the probe session
        session.get_items = mocker.Mock(return_value=mock_samples)

        params.params = {"METHOD": "scan", "ADAPTIVE": "1"}
        bed_mesh_macro.run(params)

        # Verify adaptive mesh was applied
        assert len(adapter.mesh_positions) == len(expected_adaptive_positions)

        zero_reference_height = -0.32

        # Collect expected mesh positions (in probe coordinates)
        expected_positions = {
            Position(x=round(px), y=round(py), z=round(mesh_config.height - z - zero_reference_height, 2))
            for (px, py), z in zip(expected_adaptive_positions, heights)
        }

        # Collect actual mesh positions (round to avoid float precision errors)
        actual_positions = {Position(x=round(p.x), y=round(p.y), z=round(p.z, 2)) for p in adapter.mesh_positions}

        assert actual_positions == expected_positions

        # Verify that no profile was set for adaptive mesh
        assert adapter.mesh_profile == ""

        # Convert toolhead moves to set of (rounded) XY positions
        actual_move_positions = {(round(x), round(y)) for x, y in toolhead.moves}

        # Convert expected probe positions to expected nozzle positions
        expected_nozzle_positions = {
            (round(px - probe_offset.x), round(py - probe_offset.y)) for (px, py) in expected_adaptive_positions
        }

        # Check that all expected nozzle positions were visited
        missing = expected_nozzle_positions - actual_move_positions
        assert not missing, f"Missing expected nozzle moves: {missing}"
