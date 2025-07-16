from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from cartographer.interfaces.printer import Position, Sample
from cartographer.macros.bed_mesh.helpers import (
    AdaptiveMeshCalculator,
    CoordinateTransformer,
    GridPointResult,
    MeshBounds,
    MeshGrid,
    SampleProcessor,
)


class TestMeshGrid:
    """Test cases for MeshGrid class."""

    def test_mesh_grid_creation(self):
        """Test basic mesh grid creation."""
        grid = MeshGrid(min_point=(0.0, 0.0), max_point=(10.0, 10.0), x_resolution=5, y_resolution=3)

        assert grid.min_point == (0.0, 0.0)
        assert grid.max_point == (10.0, 10.0)
        assert grid.x_resolution == 5
        assert grid.y_resolution == 3

    def test_mesh_grid_validation(self):
        """Test that grid validates minimum resolution."""
        with pytest.raises(ValueError, match="Grid resolution must be at least 3x3"):
            _ = MeshGrid((0.0, 0.0), (10.0, 10.0), 2, 3)

        with pytest.raises(ValueError, match="Grid resolution must be at least 3x3"):
            _ = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 2)

    def test_coordinate_properties(self):
        """Test coordinate array generation."""
        grid = MeshGrid((0.0, 0.0), (10.0, 20.0), 3, 5)

        np.testing.assert_array_equal(grid.x_coords, [0.0, 5.0, 10.0])
        np.testing.assert_array_equal(grid.y_coords, [0.0, 5.0, 10.0, 15.0, 20.0])

        assert grid.x_step == 5.0
        assert grid.y_step == 5.0

    def test_generate_points(self):
        """Test point generation in correct order."""
        grid = MeshGrid((0.0, 0.0), (2.0, 2.0), 3, 3)
        points = grid.generate_points()

        expected_points = [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (2.0, 2.0),
        ]

        assert points == expected_points

    def test_contains_point(self):
        """Test point containment checking."""
        grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)

        assert grid.contains_point((5.0, 5.0)) is True
        assert grid.contains_point((0.0, 0.0)) is True
        assert grid.contains_point((10.0, 10.0)) is True
        assert grid.contains_point((-1.0, 5.0)) is False
        assert grid.contains_point((5.0, 11.0)) is False

    def test_point_to_grid_index(self):
        """Test point to grid index conversion."""
        grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)

        assert grid.point_to_grid_index((0.0, 0.0)) == (0, 0)
        assert grid.point_to_grid_index((5.0, 5.0)) == (1, 1)
        assert grid.point_to_grid_index((10.0, 10.0)) == (2, 2)
        assert grid.point_to_grid_index((2.6, 7.6)) == (2, 1)  # Rounded

    def test_grid_index_to_point(self):
        """Test grid index to point conversion."""
        grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)

        assert grid.grid_index_to_point(0, 0) == (0.0, 0.0)
        assert grid.grid_index_to_point(1, 1) == (5.0, 5.0)
        assert grid.grid_index_to_point(2, 2) == (10.0, 10.0)

    def test_is_valid_index(self):
        """Test grid index validation."""
        grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)

        assert grid.is_valid_index(0, 0) is True
        assert grid.is_valid_index(2, 2) is True
        assert grid.is_valid_index(1, 1) is True
        assert grid.is_valid_index(-1, 0) is False
        assert grid.is_valid_index(0, -1) is False
        assert grid.is_valid_index(3, 0) is False
        assert grid.is_valid_index(0, 3) is False


def mock_sample(position: Position | None):
    return Sample(0, 0, position, None, 0)


grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)
processor = SampleProcessor(grid, max_distance=1.0)


class TestSampleProcessor:
    """Test cases for SampleProcessor class."""

    def test_assign_samples_basic(self):
        """Test basic sample assignment."""
        samples = [
            mock_sample(Position(0.0, 0.0, 0.0)),
            mock_sample(Position(5.0, 5.0, 0.0)),
            mock_sample(Position(10.0, 10.0, 0.0)),
        ]

        height_calc = Mock(return_value=2.5)
        results = processor.assign_samples_to_grid(samples, height_calc)

        assert len(results) == 9  # 3x3 grid
        assert all(isinstance(r, GridPointResult) for r in results)

        # Check that height calculation was called for each valid sample
        assert height_calc.call_count == 3

    def test_assign_samples_with_none_position(self):
        """Test sample assignment with None positions."""
        samples = [mock_sample(None), mock_sample(Position(5.0, 5.0, 0.0))]

        height_calc = Mock(return_value=2.5)
        results = processor.assign_samples_to_grid(samples, height_calc)

        assert len(results) == 9
        assert height_calc.call_count == 1  # Only one valid sample

    def test_assign_samples_out_of_bounds(self):
        """Test sample assignment with out-of-bounds samples."""
        samples = [
            mock_sample(Position(-5.0, 5.0, 0.0)),  # Out of bounds
            mock_sample(Position(15.0, 5.0, 0.0)),  # Out of bounds
            mock_sample(Position(5.0, 5.0, 0.0)),  # In bounds
        ]

        height_calc = Mock(return_value=2.5)
        results = processor.assign_samples_to_grid(samples, height_calc)

        assert len(results) == 9
        assert height_calc.call_count == 1  # Only one valid sample

    def test_assign_samples_max_distance(self):
        """Test sample assignment respects max distance."""
        # Create a processor with very small max distance
        strict_processor = SampleProcessor(grid, max_distance=0.1)

        samples = [
            mock_sample(Position(0.5, 0.5, 0.0)),  # Too far from (0,0)
            mock_sample(Position(0.05, 0.05, 0.0)),  # Close to (0,0)
        ]

        height_calc = Mock(return_value=2.5)
        results = strict_processor.assign_samples_to_grid(samples, height_calc)

        assert len(results) == 9
        assert height_calc.call_count == 1  # Only one sample within distance

    def test_assign_samples_median_calculation(self):
        """Test that median is calculated correctly for multiple samples."""
        # Create multiple samples at the same grid point
        samples = [
            mock_sample(Position(0.0, 0.0, 0.0)),
            mock_sample(Position(0.0, 0.0, 0.0)),
            mock_sample(Position(0.0, 0.0, 0.0)),
        ]

        # Mock different heights for each sample
        height_calc = Mock(side_effect=[1.0, 2.0, 3.0])

        results = processor.assign_samples_to_grid(samples, height_calc)

        # Find the result for point (0,0)
        result_00 = next(r for r in results if r.point == (0.0, 0.0))
        assert result_00.z == 2.0  # Median of [1.0, 2.0, 3.0]
        assert result_00.sample_count == 3


transformer = CoordinateTransformer(probe_offset=Position(2.0, 1.0, 0))


class TestCoordinateTransformer:
    """Test cases for CoordinateTransformer class."""

    def test_probe_to_nozzle_conversion(self):
        """Test probe to nozzle coordinate conversion."""
        result = transformer.probe_to_nozzle((10.0, 5.0))
        assert result == (8.0, 4.0)

    def test_nozzle_to_probe_conversion(self):
        """Test nozzle to probe coordinate conversion."""
        result = transformer.nozzle_to_probe((8.0, 4.0))
        assert result == (10.0, 5.0)

    def test_coordinate_conversion_roundtrip(self):
        """Test that coordinate conversions are reversible."""
        original = (10.0, 5.0)
        nozzle = transformer.probe_to_nozzle(original)
        back_to_probe = transformer.nozzle_to_probe(nozzle)

        assert back_to_probe == original

    def test_normalize_to_zero_reference_simple(self):
        """Test zero reference normalization with simple case."""
        positions = [Position(0.0, 0.0, 1.0), Position(1.0, 0.0, 2.0), Position(0.0, 1.0, 1.5), Position(1.0, 1.0, 2.5)]

        zero_ref = (0.0, 0.0)
        normalized = transformer.normalize_to_zero_reference(positions, zero_ref)

        # The point at (0,0) should become z=0
        normalized_00 = next(p for p in normalized if p.x == 0.0 and p.y == 0.0)
        assert abs(normalized_00.z) < 1e-10  # Should be very close to 0


class TestMeshBounds:
    """Test cases for MeshBounds class."""

    def test_mesh_bounds_creation(self):
        """Test mesh bounds creation."""
        bounds = MeshBounds((0.0, 0.0), (10.0, 20.0))

        assert bounds.min_point == (0.0, 0.0)
        assert bounds.max_point == (10.0, 20.0)

    def test_width_calculation(self):
        """Test width calculation."""
        bounds = MeshBounds((0.0, 0.0), (10.0, 20.0))
        assert bounds.width() == 10.0

    def test_height_calculation(self):
        """Test height calculation."""
        bounds = MeshBounds((0.0, 0.0), (10.0, 20.0))
        assert bounds.height() == 20.0


base_bounds = MeshBounds((0.0, 0.0), (100.0, 100.0))
base_resolution = (11, 11)
calculator = AdaptiveMeshCalculator(base_bounds, base_resolution)


class TestAdaptiveMeshCalculator:
    """Test cases for AdaptiveMeshCalculator class."""

    def test_adaptive_bounds_no_objects(self):
        """Test adaptive bounds with no objects."""
        result = calculator.calculate_adaptive_bounds([], 5.0)
        assert result == base_bounds

    def test_adaptive_bounds_with_objects(self):
        """Test adaptive bounds with objects."""
        object_points = [(20.0, 20.0), (30.0, 30.0), (40.0, 25.0)]
        margin = 5.0

        result = calculator.calculate_adaptive_bounds(object_points, margin)

        # Expected bounds: min object point - margin, max object point + margin
        # But clamped to base bounds
        expected_min = (15.0, 15.0)  # min(20) - 5, min(20) - 5
        expected_max = (45.0, 35.0)  # max(40) + 5, max(30) + 5

        assert result.min_point == expected_min
        assert result.max_point == expected_max

    def test_adaptive_bounds_clamping(self):
        """Test that adaptive bounds are clamped to base bounds."""
        object_points = [(-10.0, -10.0), (110.0, 110.0)]
        margin = 5.0

        result = calculator.calculate_adaptive_bounds(object_points, margin)

        # Should be clamped to base bounds
        assert result.min_point == (0.0, 0.0)
        assert result.max_point == (100.0, 100.0)

    def test_adaptive_resolution_same_size(self):
        """Test adaptive resolution with same size bounds."""
        adaptive_bounds = base_bounds
        result = calculator.calculate_adaptive_resolution(adaptive_bounds)

        assert result == base_resolution

    def test_adaptive_resolution_half_size(self):
        """Test adaptive resolution with half-size bounds."""
        adaptive_bounds = MeshBounds((25.0, 25.0), (75.0, 75.0))
        result = calculator.calculate_adaptive_resolution(adaptive_bounds)

        # Half the width/height should give roughly half the resolution
        # But at least 3 points minimum
        expected_x = max(3, int(50.0 * (10.0 / 100.0)) + 1)  # 6
        expected_y = max(3, int(50.0 * (10.0 / 100.0)) + 1)  # 6

        assert result == (expected_x, expected_y)

    def test_adaptive_resolution_minimum_points(self):
        """Test that adaptive resolution respects minimum points."""
        # Very small adaptive bounds
        adaptive_bounds = MeshBounds((49.0, 49.0), (51.0, 51.0))
        result = calculator.calculate_adaptive_resolution(adaptive_bounds)

        # Should be at least 3x3
        assert result[0] >= 3
        assert result[1] >= 3


# Integration tests
class TestIntegration:
    """Integration tests for the helper classes working together."""

    def test_complete_workflow(self):
        """Test a complete workflow from grid creation to sample processing."""
        # Create grid
        grid = MeshGrid((0.0, 0.0), (10.0, 10.0), 3, 3)

        # Create sample processor
        processor = SampleProcessor(grid, max_distance=1.0)

        # Create mock samples
        samples = [
            mock_sample(Position(0.0, 0.0, 0.0)),
            mock_sample(Position(5.0, 5.0, 0.0)),
            mock_sample(Position(10.0, 10.0, 0.0)),
        ]

        # Mock height calculation
        height_calc = Mock(return_value=2.5)

        # Process samples
        results = processor.assign_samples_to_grid(samples, height_calc)

        # Verify results
        assert len(results) == 9  # 3x3 grid
        assert all(isinstance(r, GridPointResult) for r in results)

        # Check that some results have samples
        sample_counts = [r.sample_count for r in results]
        assert sum(sample_counts) == 3  # Total samples assigned

    def test_adaptive_workflow(self):
        """Test adaptive mesh calculation workflow."""
        # Create base configuration
        base_bounds = MeshBounds((0.0, 0.0), (100.0, 100.0))
        base_resolution = (11, 11)

        # Create calculator
        calculator = AdaptiveMeshCalculator(base_bounds, base_resolution)

        # Define object points
        object_points = [(20.0, 20.0), (30.0, 30.0)]

        # Calculate adaptive bounds and resolution
        adaptive_bounds = calculator.calculate_adaptive_bounds(object_points, 5.0)
        adaptive_resolution = calculator.calculate_adaptive_resolution(adaptive_bounds)

        # Create adaptive grid
        grid = MeshGrid(
            adaptive_bounds.min_point, adaptive_bounds.max_point, adaptive_resolution[0], adaptive_resolution[1]
        )

        # Verify grid properties
        assert grid.min_point == (15.0, 15.0)
        assert grid.max_point == (35.0, 35.0)
        assert grid.x_resolution >= 3
        assert grid.y_resolution >= 3

    def test_coordinate_transform_workflow(self):
        """Test coordinate transformation workflow."""
        # Create transformer
        transformer = CoordinateTransformer(probe_offset=Position(2.0, 1.0, 0))

        # Create some positions
        positions = [Position(0.0, 0.0, 1.0), Position(0.0, 2.0, 1.0), Position(2.0, 0.0, 2.0), Position(2.0, 2.0, 2.0)]

        # Normalize to zero reference
        normalized = transformer.normalize_to_zero_reference(positions, (0.0, 0.0))

        # Verify normalization
        assert len(normalized) == 4

        # First position should have z ≈ 0
        first_pos = next(p for p in normalized if p.x == 0.0 and p.y == 0.0)
        assert abs(first_pos.z) < 1e-10
