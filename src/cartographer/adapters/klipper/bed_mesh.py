from __future__ import annotations

from typing import TYPE_CHECKING, final

import numpy as np
from extras import bed_mesh
from typing_extensions import override

from cartographer.macros.bed_mesh.interfaces import BedMeshAdapter, Polygon

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from extras.bed_mesh import _Params as BedMeshParams  # pyright: ignore[reportPrivateUsage]

    from cartographer.interfaces.printer import Position

ROUND_DECIMALS = 2


@final
class KlipperBedMesh(BedMeshAdapter):
    def __init__(self, config: ConfigWrapper) -> None:
        self.config = config.getsection("bed_mesh")
        self.bed_mesh = config.get_printer().load_object(self.config, "bed_mesh")
        self.printer = config.get_printer()

    @override
    def get_objects(self) -> list[Polygon]:
        exclude_object = self.printer.lookup_object("exclude_object", None)
        if not exclude_object:
            return []

        objects = exclude_object.get_status().get("objects", [])
        polygons: list[Polygon] = []

        for obj in objects:
            polygon = obj.get("polygon", [])
            points = [(x, y) for x, y in polygon]
            polygons.append(points)

        return polygons

    @override
    def clear_mesh(self) -> None:
        self.bed_mesh.set_mesh(None)

    @override
    def apply_mesh(self, mesh_points: list[Position], profile_name: str | None = None) -> None:
        coords = np.asarray([p.as_tuple() for p in mesh_points])

        xs_rounded = np.round(coords[:, 0], ROUND_DECIMALS)
        ys_rounded = np.round(coords[:, 1], ROUND_DECIMALS)
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

        mesh_params: BedMeshParams = {
            "min_x": float(x_unique[0]),
            "min_y": float(y_unique[0]),
            "max_x": float(x_unique[-1]),
            "max_y": float(y_unique[-1]),
            "x_count": points_per_x,
            "y_count": points_per_y,
            "mesh_x_pps": 0,
            "mesh_y_pps": 0,
            "algo": "bicubic",
            "tension": 0.2,
        }

        mesh = bed_mesh.ZMesh(mesh_params, profile_name)
        try:
            native_matrix = [[float(z) for z in row] for row in matrix.tolist()]
            mesh.build_mesh(native_matrix)
        except bed_mesh.BedMeshError as e:
            raise RuntimeError(str(e)) from e

        self.bed_mesh.set_mesh(mesh)
        if profile_name is not None:
            self.bed_mesh.save_profile(profile_name)
