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
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator

if TYPE_CHECKING:
    from matplotlib.text import Annotation

    from cartographer.macros.bed_mesh.interfaces import PathGenerator, Point


def make_grid(nx: int, ny: int, spacing: float) -> list[Point]:
    return [(1 + x * spacing, 1 + y * spacing) for y in range(ny) for x in range(nx)]


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


def animate_paths(generator: PathGenerator, grid_shapes: list[tuple[int, int]], spacing=5, interval=300):
    fig, axes = plt.subplots(nrows=(len(grid_shapes) + 2) // 3, ncols=3, figsize=(16, 10))
    axes = axes.flatten()

    all_grids: list[list[Point]] = []
    all_paths: list[list[Point]] = []
    arrows: list[Annotation] = []

    # Prepare plots & arrows
    for ax, (nx, ny) in zip(axes, grid_shapes):
        grid = make_grid(nx, ny, spacing)
        path = list(generator.generate_path(grid, (0, nx * spacing + 2), (0, ny * spacing + 2)))
        plot_grid_and_path(ax, f"{nx}×{ny} grid", grid, path)
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

    for ax in axes[len(grid_shapes) :]:
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
    "spiral": SpiralPathGenerator,
    "random": RandomPathGenerator,
}

if __name__ == "__main__":
    path_strategy_type = sys.argv[1]
    path_strategy = PATH_STRATEGY_MAP.get(path_strategy_type)
    if path_strategy is None:
        msg = f"Unknown path strategy: {path_strategy_type}"
        raise ValueError(msg)
    generator = path_strategy("x")

    grid_shapes = [
        (3, 3),
        (4, 4),
        (5, 3),
        (6, 4),
        (10, 11),
        (7, 11),
        (9, 8),
    ]

    animate_paths(generator, grid_shapes)
