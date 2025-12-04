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
    Region,
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
    return Sample(0, 0, position, 0, 0)


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
        normalized = transformer.normalize_to_zero_reference_point(positions, zero_ref=zero_ref)

        # The point at (0,0) should become z=0
        normalized_00 = next(p for p in normalized if p.x == 0.0 and p.y == 0.0)
        assert abs(normalized_00.z) < 1e-10  # Should be very close to 0

    def test_small_grid_with_interpolation(self):
        positions = [
            Position(0, 0, 1.0),
            Position(1, 0, 2.0),
            Position(0, 1, 3.0),
            Position(1, 1, 4.0),
        ]
        zero_ref = (0.5, 0.5)  # Middle point, bilinear = 2.5
        result = transformer.normalize_to_zero_reference_point(positions, zero_ref=zero_ref)
        z_values = [p.z for p in result]
        assert all(abs(z - expected) < 1e-9 for z, expected in zip(z_values, [-1.5, -0.5, 0.5, 1.5]))

    def test_grid_with_explicit_height(self):
        positions = [Position(x, y, float(x + y)) for y in range(3) for x in range(3)]
        zero_height = 2.0
        result = transformer.normalize_to_zero_reference_point(positions, zero_height=zero_height)
        z_values = [p.z for p in result]
        assert all(abs(z - ((p.x + p.y) - 2.0)) < 1e-9 for p, z in zip(positions, z_values))

    def test_zero_reference_at_edge(self):
        """Test that interpolation handles zero reference points at grid edges properly."""
        transformer = CoordinateTransformer(probe_offset=Position(0.0, 0.0, 0.0))

        # Create a 4x4 grid of positions
        positions: list[Position] = []
        for y in range(4):
            for x in range(4):
                positions.append(Position(x, y, float(x + y)))

        result = transformer.normalize_to_zero_reference_point(positions, zero_ref=(3.0, 3.0))

        # Verify the normalization was done correctly
        # The position at (3,3) should have z=0 after normalization
        edge_pos = next(p for p in result if p.x == 3.0 and p.y == 3.0)
        assert abs(edge_pos.z) < 1e-10, "Zero reference point should have z≈0"

        # Check a different point to ensure values were properly adjusted
        # For example, position (0,0) should be 0 - 6 = -6
        origin_pos = next(p for p in result if p.x == 0.0 and p.y == 0.0)
        assert abs(origin_pos.z + 6.0) < 1e-10, "Points should be normalized relative to reference"

    @pytest.mark.parametrize("corner", [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)])
    def test_zero_reference_at_corner(self, corner: tuple[float, float]):
        """Test interpolation with zero reference at all four corners."""
        transformer = CoordinateTransformer(probe_offset=Position(0.0, 0.0, 0.0))

        # Create a 4x4 grid of positions
        positions: list[Position] = []
        for y in range(4):
            for x in range(4):
                positions.append(Position(x, y, float(x + y)))

        result = transformer.normalize_to_zero_reference_point(positions, zero_ref=corner)

        # The corner point should have z=0
        corner_pos = next(p for p in result if p.x == corner[0] and p.y == corner[1])
        assert abs(corner_pos.z) < 1e-10, f"Corner {corner} should have z≈0"

    def test_all_same_height(self):
        positions = [Position(x, y, 5.0) for y in range(2) for x in range(2)]
        result = transformer.normalize_to_zero_reference_point(positions, zero_height=5.0)
        assert all(p.z == 0.0 for p in result)

    def _create_test_grid(self, faulty_regions: list[Region] | None = None, size: int = 4) -> list[Position]:
        """Create a simple grid of positions with optional faulty regions assigned a distinct z-value."""
        positions: list[Position] = []
        for y in range(size):
            for x in range(size):
                z = x + y
                if faulty_regions:
                    for region in faulty_regions:
                        if region.contains_point((x, y)):
                            z = -1000  # clearly wrong value
                positions.append(Position(x, y, z))
        return positions

    def test_single_faulty_point_rbf(self):
        region = Region(min_point=(1, 1), max_point=(1, 1))
        positions = self._create_test_grid(faulty_regions=[region])

        output = transformer.apply_faulty_regions(positions, faulty_regions=[region])

        for p_old, p_new in zip(positions, output):
            pt = (p_old.x, p_old.y)
            if region.contains_point(pt):
                assert p_new.z != -1000  # must be replaced
            else:
                assert p_new.z == p_old.z  # unchanged

    def test_multiple_faulty_points_rbf(self):
        regions = [Region(min_point=(0, 0), max_point=(1, 1)), Region(min_point=(2, 2), max_point=(3, 3))]
        positions = self._create_test_grid(faulty_regions=regions)

        output = transformer.apply_faulty_regions(positions, faulty_regions=regions)

        for p_old, p_new in zip(positions, output):
            pt = (p_old.x, p_old.y)
            if any(region.contains_point(pt) for region in regions):
                assert p_new.z != -1000  # replaced by RBF
            else:
                assert p_new.z == p_old.z  # unchanged

    def test_large_bed_mesh_with_faulty_regions(self):
        """Test interpolation on real 25x23 Klipper bed mesh with four faulty regions."""

        # Mesh values (23 rows × 25 cols)
        # fmt: off
        mesh_values = [
            [-0.277889, -0.230035, -0.208624, -0.189581, -0.176448, -0.152713, -0.131865, -0.111034, -0.100966, -0.091873, -0.092057, -0.095770, -0.093169, -0.089465, -0.089282, -0.100966, -0.104691, -0.120203, -0.135459, -0.147390, -0.157860, -0.180308, -0.205711, -0.236749, -0.267014],  # noqa: E501
            [-0.267014, -0.216281, -0.190936, -0.175102, -0.156529, -0.134322, -0.108606, -0.094466, -0.083006, -0.073065, -0.074350, -0.074168, -0.071782, -0.071780, -0.071596, -0.080605, -0.087062, -0.104691, -0.115145, -0.130545, -0.135828, -0.177795, -0.188226, -0.220579, -0.250252],  # noqa: E501

            [-0.255825, -0.212355, -0.206885, -0.258620, -0.193649, -0.127909, -0.107300, -0.079498, -0.066644, -0.059332, -0.058971, -0.056780, -0.054231, -0.056420, -0.059332, -0.064084, -0.073986, -0.093353, -0.104691, -0.120019, -0.139614, -0.239521, -0.257223, -0.219206, -0.250438], # noqa: E501
            [-0.244687, -0.199080, -0.199080, -0.242286, -0.199080, -0.104691, -0.082077, -0.061526, -0.044064, -0.039356, -0.039356, -0.036464, -0.036827, -0.034301, -0.041524, -0.044064, -0.051685, -0.076558, -0.084293, -0.102449, -0.122646, -0.247839, -0.255825, -0.209618, -0.236747], # noqa: E501
            [-0.243486, -0.188413, -0.173755, -0.156343, -0.130547, -0.099665, -0.074350, -0.054233, -0.040259, -0.031778, -0.029252, -0.031778, -0.029252, -0.024581, -0.028891, -0.036464, -0.046601, -0.064451, -0.079498, -0.096884, -0.112527, -0.146425, -0.177795, -0.207260, -0.239517], # noqa: E501
            [-0.242286, -0.188226, -0.156708, -0.138105, -0.115145, -0.086878, -0.066644, -0.046603, -0.033039, -0.024221, -0.025478, -0.024401, -0.022965, -0.023145, -0.023145, -0.034121, -0.041528, -0.060429, -0.070497, -0.090761, -0.098546, -0.138107, -0.159379, -0.201435, -0.232785], # noqa: E501
            [-0.238132, -0.183004, -0.167045, -0.131865, -0.108606, -0.090578, -0.067929, -0.044243, -0.027994, -0.024221, -0.025478, -0.016689, -0.015617, -0.018307, -0.019195, -0.024221, -0.032858, -0.054052, -0.070497, -0.081714, -0.099665, -0.129227, -0.155380, -0.192292, -0.236560], # noqa: E501
            [-0.242286, -0.189581, -0.160717, -0.130363, -0.107300, -0.086878, -0.061528, -0.036647, -0.026735, -0.015800, -0.018307, -0.015620, -0.010617, -0.013116, -0.016872, -0.021709, -0.029252, -0.049142, -0.061526, -0.081712, -0.089465, -0.125643, -0.148719, -0.193276, -0.228462], # noqa: E501
            [-0.247839, -0.183187, -0.164732, -0.133185, -0.112527, -0.084293, -0.064084, -0.039356, -0.026735, -0.014545, -0.012045, -0.014545, -0.009547, -0.007052, -0.011686, -0.017054, -0.026735, -0.036827, -0.059152, -0.076922, -0.090577, -0.116455, -0.147572, -0.189581, -0.235363], # noqa: E501
            [-0.250252, -0.193649, -0.156708, -0.133185, -0.104691, -0.082077, -0.061526, -0.041524, -0.024581, -0.016689, -0.017054, -0.011686, -0.009547, -0.007052, -0.009547, -0.019560, -0.022069, -0.044423, -0.054231, -0.074350, -0.086878, -0.120019, -0.148719, -0.193649, -0.228835], # noqa: E501
            [-0.243302, -0.180313, -0.160717, -0.125277, -0.100785, -0.079317, -0.059153, -0.030515, -0.017942, -0.014186, -0.007052, -0.007052, 0.000420, -0.002428, -0.004555, -0.007052, -0.019560, -0.041888, -0.054231, -0.069213, -0.086878, -0.117766, -0.151381, -0.188226, -0.233980], # noqa: E501
            [-0.245063, -0.188226, -0.151381, -0.125276, -0.099484, -0.073986, -0.054231, -0.029256, -0.016689, -0.008300, -0.008300, 0.000420, 0.000421, 0.001306, 0.000064, -0.010796, -0.016689, -0.035564, -0.051686, -0.070497, -0.082077, -0.116455, -0.147390, -0.192109, -0.236747], # noqa: E501
            [-0.250252, -0.185524, -0.157866, -0.126592, -0.104691, -0.075454, -0.056780, -0.033040, -0.015620, -0.008300, -0.004559, -0.005804, -0.002066, 0.006268, 0.002909, -0.009369, -0.021708, -0.038091, -0.052778, -0.074350, -0.088172, -0.112533, -0.150050, -0.193649, -0.242101], # noqa: E501
            [-0.253034, -0.189581, -0.155377, -0.126592, -0.100966, -0.078028, -0.054052, -0.030334, -0.016691, -0.008297, -0.007052, -0.004559, 0.000064, 0.004149, 0.000421, -0.007051, -0.019195, -0.041524, -0.051685, -0.074350, -0.084293, -0.120019, -0.148719, -0.196366, -0.236747], # noqa: E501
            [-0.261412, -0.199080, -0.175101, -0.135828, -0.109912, -0.089465, -0.066644, -0.041888, -0.026735, -0.019560, -0.014186, -0.009547, -0.009547, -0.009547, -0.007052, -0.011686, -0.026735, -0.051325, -0.066649, -0.080423, -0.098184, -0.129043, -0.161872, -0.200261, -0.249045], # noqa: E501
            [-0.269448, -0.206885, -0.167045, -0.141123, -0.112527, -0.092056, -0.066644, -0.044064, -0.029252, -0.019560, -0.019195, -0.016689, -0.014186, -0.009547, -0.012045, -0.019195, -0.031778, -0.046601, -0.064084, -0.084293, -0.096884, -0.133185, -0.164364, -0.206885, -0.247468], # noqa: E501
            [-0.273662, -0.207260, -0.182821, -0.148721, -0.122830, -0.098184, -0.075454, -0.052958, -0.040259, -0.024221, -0.024221, -0.024221, -0.021709, -0.016689, -0.021709, -0.029252, -0.036827, -0.053871, -0.073986, -0.094283, -0.109912, -0.133185, -0.169723, -0.209988, -0.258620], # noqa: E501
            [-0.272629, -0.212355, -0.177424, -0.148719, -0.120385, -0.102086, -0.079134, -0.052958, -0.035567, -0.031596, -0.029261, -0.025658, -0.021889, -0.023145, -0.026735, -0.031596, -0.042794, -0.066644, -0.079498, -0.097246, -0.116455, -0.151382, -0.178958, -0.223141, -0.261412], # noqa: E501
            [-0.277889, -0.216461, -0.190937, -0.155194, -0.130545, -0.108606, -0.085586, -0.056420, -0.044064, -0.040440, -0.039174, -0.031781, -0.025658, -0.030518, -0.031778, -0.036647, -0.050233, -0.073065, -0.089284, -0.105996, -0.122646, -0.155377, -0.188413, -0.227273, -0.273850], # noqa: E501
            [-0.286357, -0.227273, -0.191122, -0.165888, -0.138289, -0.113836, -0.089467, -0.066644, -0.054051, -0.042794, -0.041888, -0.039176, -0.034301, -0.034121, -0.035383, -0.045333, -0.056780, -0.079134, -0.096884, -0.112527, -0.130545, -0.164732, -0.196366, -0.236747, -0.272254], # noqa: E501
            [-0.292022, -0.231222, -0.215089, -0.236747, -0.185524, -0.125276, -0.102449, -0.079134, -0.064084, -0.051685, -0.051325, -0.049142, -0.044423, -0.039356, -0.046601, -0.056420, -0.066644, -0.086878, -0.108606, -0.128094, -0.146062, -0.228469, -0.260019, -0.244875, -0.283533], # noqa: E501
            [-0.294485, -0.242286, -0.233980, -0.245063, -0.233980, -0.140754, -0.115145, -0.089465, -0.074350, -0.069213, -0.066644, -0.061526, -0.058971, -0.061526, -0.064084, -0.071412, -0.081712, -0.107300, -0.123012, -0.140754, -0.164364, -0.280706, -0.314454, -0.253034, -0.283533], # noqa: E501
            [-0.300361, -0.242101, -0.228463, -0.215089, -0.187058, -0.152713, -0.133187, -0.104874, -0.089465, -0.079134, -0.074350, -0.071780, -0.069213, -0.069213, -0.074350, -0.079134, -0.094283, -0.120019, -0.138105, -0.154045, -0.169723, -0.212355, -0.236747, -0.253034, -0.291650], # noqa: E501
        ] # fmt: on

        # Grid spacing
        x_count, y_count = 25, 23
        min_x, max_x = 0.0, 250.0
        min_y, max_y = 20.0, 250.0
        x_step = (max_x - min_x) / (x_count - 1)
        y_step = (max_y - min_y) / (y_count - 1)

        # Define faulty regions in real printer coordinates
        faulty_regions = [
            Region((15.0, 25.0), (55.0, 65.0)),
            Region((205.0, 25.0), (245.0, 65.0)),
            Region((15.0, 215.0), (55.0, 250.0)),
            Region((205.0, 215.0), (245.0, 250.0)),
        ]

        # Build positions
        positions: list[Position] = []
        for j, row in enumerate(mesh_values):
            y = min_y + j * y_step
            for i, z in enumerate(row):
                x = min_x + i * x_step
                positions.append(Position(x, y, z))

        # Interpolate faulty regions
        output = transformer.apply_faulty_regions(positions, faulty_regions=faulty_regions)

        # Assertions
        for p_old, p_new in zip(positions, output):
            pt = (p_old.x, p_old.y)
            if any(region.contains_point(pt) for region in faulty_regions):
                # Value must have been replaced, not equal to original
                assert abs(p_new.z - p_old.z) > 1e-9
            else:
                # Outside faulty regions, must remain identical
                assert p_new.z == p_old.z


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
        normalized = transformer.normalize_to_zero_reference_point(positions, zero_ref=(0.0, 0.0))

        # Verify normalization
        assert len(normalized) == 4

        # First position should have z ≈ 0
        first_pos = next(p for p in normalized if p.x == 0.0 and p.y == 0.0)
        assert abs(first_pos.z) < 1e-10
