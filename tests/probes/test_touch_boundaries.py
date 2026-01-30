from __future__ import annotations

import pytest

from cartographer.probe.touch_mode import TouchBoundaries, TouchModeConfiguration

BOUNDS = TouchBoundaries(min_x=10.0, max_x=20.0, min_y=5.0, max_y=15.0)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        # Clearly within bounds
        (15.0, 10.0, True),
        (10.0, 5.0, True),  # on lower edge
        (20.0, 15.0, True),  # on upper edge
        # Clearly outside bounds (beyond tolerance)
        (9.98, 10.0, False),
        (20.02, 10.0, False),
        (15.0, 4.98, False),
        (15.0, 15.02, False),
        # Near boundary (within 0.01 tolerance)
        (9.99, 10.0, True),
        (20.01, 10.0, True),
        (15.0, 4.99, True),
        (15.0, 15.01, True),
    ],
)
def test_is_within_bounds(x: float, y: float, expected: bool) -> None:
    assert BOUNDS.is_within(x=x, y=y) is expected


def make_config(
    mesh_min: tuple[float, float],
    mesh_max: tuple[float, float],
    x_offset: float,
    y_offset: float,
) -> TouchModeConfiguration:
    return TouchModeConfiguration(
        samples=1,
        max_samples=1,
        mesh_min=mesh_min,
        mesh_max=mesh_max,
        max_touch_temperature=150,
        x_offset=x_offset,
        y_offset=y_offset,
        lift_speed=5,
        retract_distance=2.0,
        models={},
    )


@pytest.mark.parametrize(
    "mesh_min, mesh_max, x_offset, y_offset, expected",
    [
        # Zero offsets
        ((0.0, 0.0), (100.0, 100.0), 0.0, 0.0, TouchBoundaries(min_x=0.0, max_x=100.0, min_y=0.0, max_y=100.0)),
        # Positive offsets
        ((0.0, 0.0), (100.0, 100.0), 10.0, 5.0, TouchBoundaries(min_x=0.0, max_x=90.0, min_y=0.0, max_y=95.0)),
        # Negative offsets
        ((0.0, 0.0), (100.0, 100.0), -10.0, -5.0, TouchBoundaries(min_x=10.0, max_x=100.0, min_y=5.0, max_y=100.0)),
        # Non-zero mesh_min
        ((10.0, 20.0), (100.0, 100.0), 5.0, 10.0, TouchBoundaries(min_x=10.0, max_x=95.0, min_y=20.0, max_y=90.0)),
    ],
)
def test_from_config_bounds(
    mesh_min: tuple[float, float],
    mesh_max: tuple[float, float],
    x_offset: float,
    y_offset: float,
    expected: TouchBoundaries,
) -> None:
    config = make_config(mesh_min, mesh_max, x_offset, y_offset)
    bounds = TouchBoundaries.from_config(config)
    assert bounds == expected
