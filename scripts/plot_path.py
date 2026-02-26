# pyright: reportUnknownMemberType=false
# pyright: reportUnusedCallResult=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.circular_snake_path import CircularSnakePathGenerator
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator

if TYPE_CHECKING:
    from matplotlib.text import Annotation

    from cartographer.macros.bed_mesh.interfaces import PathGenerator, Point


def make_grid(nx: int, ny: int, spacing: float) -> list[Point]:
    return [(1 + x * spacing, 1 + y * spacing) for y in range(ny) for x in range(nx)]


def make_circular_grid(radius: float, spacing: float, center=(0.0, 0.0)) -> list[Point]:
    """Create a circular grid of points within a given radius."""
    cx, cy = center
    # Determine grid bounds
    max_coord = radius
    n_points = int(2 * max_coord / spacing) + 1
    
    points = []
    for i in range(n_points):
        for j in range(n_points):
            x = cx + (i - n_points // 2) * spacing
            y = cy + (j - n_points // 2) * spacing
            
            # Check if point is within circle (with small epsilon for boundary)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq <= (radius + 1e-2) ** 2:
                points.append((x, y))
    
    return points


def plot_grid_and_path(ax, name: str, grid: list[Point], path: list[Point]):
    # Plot all grid points as light gray background dots
    gx, gy = zip(*grid)
    ax.scatter(gx, gy, color="red", s=20, label="Grid")

    # Plot path as blue line
    px, py = zip(*path)
    ax.plot(px, py, linestyle="-", color="blue", label="Path")

    ax.set_title(name)
    ax.set_aspect("equal")
    ax.grid(True)


def animate_paths(generator: PathGenerator, grid_configs, spacing=5, interval=300):
    """Animate path generation for both rectangular and circular grids.
    
    grid_configs can be:
    - List of (nx, ny) tuples for rectangular grids
    - List of {"type": "circular", "radius": float, "spacing": float} for circular grids
    """
    fig, axes = plt.subplots(nrows=(len(grid_configs) + 2) // 3, ncols=3, figsize=(16, 10))
    axes = axes.flatten()

    all_grids: list[list[Point]] = []
    all_paths: list[list[Point]] = []
    arrows: list[Annotation] = []

    # Prepare plots & arrows
    for ax, config in zip(axes, grid_configs):
        if isinstance(config, dict) and config.get("type") == "circular":
            # Circular grid configuration
            radius = config["radius"]
            grid_spacing = config.get("spacing", spacing)
            grid = make_circular_grid(radius, grid_spacing)
            bounds = (-radius - 5, radius + 5)
            path = list(generator.generate_path(grid, bounds, bounds))
            title = f"Circular r={radius}"
        else:
            # Rectangular grid configuration (nx, ny)
            nx, ny = config
            grid = make_grid(nx, ny, spacing)
            path = list(generator.generate_path(grid, (0, nx * spacing + 2), (0, ny * spacing + 2)))
            title = f"{nx}×{ny} grid"
            
        plot_grid_and_path(ax, title, grid, path)
        all_grids.append(grid)
        all_paths.append(path)

        # Create arrow annotation at start of path
        arrow = ax.annotate(
            "",
            xy=path[0],
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=2,
            ),
        )
        arrows.append(arrow)

    for ax in axes[len(grid_configs):]:
        ax.axis("off")

    def update(frame: int):
        for arrow, path in zip(arrows, all_paths):
            i = frame % len(path)  # Loop through frames
            arrow.set_position(xy=(float(path[i - 1][0]), float(path[i - 1][1])))
            arrow.xy = (float(path[i][0]), float(path[i][1]))
        return arrows

    _ = FuncAnimation(fig, update, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()


PATH_STRATEGY_MAP = {
    "snake": SnakePathGenerator,
    "alternating_snake": AlternatingSnakePathGenerator,
    "circular_snake": CircularSnakePathGenerator,
    "spiral": SpiralPathGenerator,
    "random": RandomPathGenerator,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_path.py <strategy> [test_type]")
        print("Strategies:", list(PATH_STRATEGY_MAP.keys()))
        print("Test types: rectangular, circular")
        sys.exit(1)
        
    path_strategy_type = sys.argv[1]
    test_type = sys.argv[2] if len(sys.argv) > 2 else "rectangular"
    
    path_strategy = PATH_STRATEGY_MAP.get(path_strategy_type)
    if path_strategy is None:
        msg = f"Unknown path strategy: {path_strategy_type}"
        raise ValueError(msg)
    generator = path_strategy("x")

    if test_type == "circular":
        # Circular mesh configurations
        grid_configs = [
            {"type": "circular", "radius": 15.0, "spacing": 5.0},
            {"type": "circular", "radius": 20.0, "spacing": 4.0},
            {"type": "circular", "radius": 25.0, "spacing": 3.0},
            {"type": "circular", "radius": 30.0, "spacing": 6.0},
        ]
        animate_paths(generator, grid_configs)
    else:
        # Rectangular grid configurations
        grid_configs = [
            (3, 3),
            (4, 4),
            (5, 3),
            (6, 4),
            (10, 11),
            (7, 11),
            (9, 8),
        ]
        animate_paths(generator, grid_configs)
