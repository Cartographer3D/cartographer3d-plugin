from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Protocol

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import numpy as np

    from cartographer.interfaces.printer import Position

Point: TypeAlias = "tuple[float|np.float_, float|np.float_]"
Polygon: TypeAlias = "list[tuple[float, float]]"


class PathGenerator(Protocol):
    def generate_path(
        self, points: list[Point], x_axis_limits: tuple[float, float], y_axis_limits: tuple[float, float]
    ) -> Iterator[Point]: ...


class BedMeshAdapter(Protocol):
    def apply_mesh(self, mesh_points: list[Position], profile_name: str | None = None): ...
    def clear_mesh(self) -> None: ...
    def get_objects(self) -> list[Polygon]: ...
