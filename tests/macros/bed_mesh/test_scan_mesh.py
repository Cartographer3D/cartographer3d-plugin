from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, final

import numpy as np
import pytest
from scipy.interpolate import griddata
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

        # Handle NaN values for circular mesh grid
        if np.isnan(matrix).any():
            # 1. Get coordinates of where we HAVE data
            valid_mask = ~np.isnan(matrix)
            # Create a coordinate grid (y, x) matching the matrix shape
            yy, xx = np.indices(matrix.shape)
            
            points_known = np.stack((yy[valid_mask], xx[valid_mask]), axis=-1)
            values_known = matrix[valid_mask]
            
            # 2. Get coordinates of where we NEED data (the NaNs)
            points_nan = np.stack((yy[~valid_mask], xx[~valid_mask]), axis=-1)

            # 3. Perform Nearest Neighbor interpolation to fill NaNs
            # This effectively "extends" the edge heights outwards to the corners
            filled_values = griddata(
                points_known, 
                values_known, 
                points_nan, 
                method='nearest'
            )

            # 4. Map the filled values back into the matrix
            matrix[~valid_mask] = filled_values

        # Final safety check after interpolation
        if np.isnan(matrix).any():
            msg = "Mesh has missing points that could not be extrapolated"
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

    @pytest.fixture
    def circular_mesh_config(self):
        """Configuration for circular bed mesh testing."""
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
            path="circular_snake",  # Spiral path works better for circular meshes
            faulty_regions=[],
            mesh_radius=40.0,
            mesh_origin=(50.0, 50.0),
            round_probe_count=5,
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

    @pytest.fixture
    def circular_bed_mesh_macro(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: BedMeshAdapter,
        circular_mesh_config: BedMeshCalibrateConfiguration,
    ):
        """Create a bed mesh macro with mocked dependencies."""
        task_executor = InlineTaskExecutor()
        macro = BedMeshCalibrateMacro(probe, toolhead, adapter, None, task_executor, circular_mesh_config)

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

    def test_circular_mesh_boundary_and_coordinate_transformation(
        self,
        mocker: MockerFixture,
        probe_offset: Position,
        params: MockParams,
        session: Mock,
        circular_mesh_config: BedMeshCalibrateConfiguration,
        circular_bed_mesh_macro: BedMeshCalibrateMacro,
        adapter: MockBedMeshAdapter,
        toolhead: MockToolhead,
    ):
        """Test that circular mesh generates only interior points and transforms coordinates properly."""

        # Calculate grid spacing from config
        mesh_min_x, mesh_min_y = circular_mesh_config.mesh_min
        mesh_max_x, mesh_max_y = circular_mesh_config.mesh_max
        probe_count_x, probe_count_y = circular_mesh_config.probe_count
        
        step_x = (mesh_max_x - mesh_min_x) / (probe_count_x - 1)
        step_y = (mesh_max_y - mesh_min_y) / (probe_count_y - 1)
        
        # Generate all grid points
        all_grid_points = [
            (float(mesh_min_x + step_x * x), float(mesh_min_y + step_y * y))
            for x in range(probe_count_x)
            for y in range(probe_count_y)
        ]
        
        # Filter to only points within circular boundary
        assert circular_mesh_config.mesh_origin is not None, "mesh_origin must be set for circular mesh"
        assert circular_mesh_config.mesh_radius is not None, "mesh_radius must be set for circular mesh"
        
        mesh_origin_x, mesh_origin_y = circular_mesh_config.mesh_origin
        mesh_radius = circular_mesh_config.mesh_radius
        
        expected_probe_positions = []
        for px, py in all_grid_points:
            distance_sq = (px - mesh_origin_x) ** 2 + (py - mesh_origin_y) ** 2
            if distance_sq <= (mesh_radius + 0.01) ** 2:  # Include epsilon for floating point
                expected_probe_positions.append((px, py))

        # Create mock heights for each position
        heights = [1.5 + 0.1 * i for i in range(len(expected_probe_positions))]

        # Create samples that would be generated when probe visits these positions
        mock_samples = self.create_samples_at_probe_positions(expected_probe_positions, heights, probe_offset)

        # Mock the probe session to return our samples
        session.get_items = mocker.Mock(return_value=mock_samples)

        params.params = {"METHOD": "scan", "PATH": "circular_snake"}
        circular_bed_mesh_macro.run(params)

        # After gap filling, mesh should have all grid points (5x5 = 25)
        # Gap filling converts sparse circular mesh to complete rectangular grid
        assert len(adapter.mesh_positions) == 25  # Full grid after gap filling
        
        # Verify all original circular mesh points are present
        mesh_coords = {(p.x, p.y) for p in adapter.mesh_positions}
        for px, py in expected_probe_positions:
            assert (px, py) in mesh_coords, f"Expected probe position ({px}, {py}) not found in mesh"
        
        # Verify corner points were filled (not NaN)
        for pos in adapter.mesh_positions:
            assert isfinite(pos.z), f"Position ({pos.x}, {pos.y}) has non-finite z value: {pos.z}"

        zero_reference_height = circular_mesh_config.height - heights[
            expected_probe_positions.index(circular_mesh_config.zero_reference_position)
        ]

        # Verify original circular mesh points have correct heights
        for i, (px, py) in enumerate(expected_probe_positions):
            expected_z = heights[i]
            
            # Find the matching mesh position
            actual_pos = None
            for pos in adapter.mesh_positions:
                if pos.x == px and pos.y == py:
                    actual_pos = pos
                    break
            
            assert actual_pos is not None, f"Position ({px}, {py}) not found in mesh"
            
            expected_mesh_z = round(circular_mesh_config.height - expected_z - zero_reference_height, 2)
            actual_mesh_z = round(actual_pos.z, 2)
            assert actual_mesh_z == expected_mesh_z, \
                f"Position ({px}, {py}): expected z={expected_mesh_z}, got z={actual_mesh_z}"

        # Verify that corner points WERE filled (gap filling adds them)
        corner_points = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0)]
        actual_xy_positions = {(round(p.x, 2), round(p.y, 2)) for p in adapter.mesh_positions}
        
        for corner in corner_points:
            assert corner in actual_xy_positions, f"Corner point {corner} should be filled by gap filling"

        # Convert toolhead moves to set of (rounded) XY positions
        actual_move_positions = {(round(x, 2), round(y, 2)) for x, y in toolhead.moves}

        # Convert expected probe positions to expected nozzle positions
        expected_nozzle_positions = {
            (round(px - probe_offset.x, 2), round(py - probe_offset.y, 2)) for (px, py) in expected_probe_positions
        }

        # Check that all expected nozzle positions were visited
        missing = expected_nozzle_positions - actual_move_positions
        assert not missing, f"Missing expected nozzle moves: {missing}"

