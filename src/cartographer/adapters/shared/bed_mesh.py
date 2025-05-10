from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from extras.bed_mesh import BedMeshError
from typing_extensions import override

from cartographer.macros.bed_mesh import MeshHelper, MeshPoint

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand, GCodeDispatch

    from cartographer.interfaces.printer import Position

logger = logging.getLogger(__name__)


@final
class KlipperMeshHelper(MeshHelper):
    def __init__(self, config: ConfigWrapper, gcode: GCodeDispatch) -> None:
        self.mesh_config = config.getsection("bed_mesh")
        self._bed_mesh = config.get_printer().load_object(self.mesh_config, "bed_mesh")
        # Loading "bed_mesh" above registers the command.
        self.macro = gcode.register_command("BED_MESH_CALIBRATE", None)

    @override
    def orig_macro(self, params: GCodeCommand) -> None:
        if self.macro is not None:
            self.macro(params)

    def _prepare(self, params: GCodeCommand) -> None:
        profile_name = params.get("PROFILE", "default")
        if not profile_name.strip():
            msg = "value for parameter 'PROFILE' must be specified"
            raise RuntimeError(msg)
        self._bed_mesh.set_mesh(None)
        self._bed_mesh.bmc._profile_name = profile_name
        try:
            self._bed_mesh.bmc.update_config(params)
        except BedMeshError as e:
            raise RuntimeError(str(e)) from e

    @override
    def prepare_scan_path(self, params: GCodeCommand) -> list[MeshPoint]:
        self._prepare(params)

        path = self._bed_mesh.bmc.probe_mgr.iter_rapid_path()
        return [MeshPoint(p[0], p[1], include) for (p, include) in path]

    @override
    def prepare_touch_points(
        self,
        params: GCodeCommand,
        *,
        mesh_min: tuple[float, float] | None,
        mesh_max: tuple[float, float] | None,
    ) -> list[MeshPoint]:
        raw = params.get_command_parameters()

        if mesh_min is not None and "MESH_MIN" not in raw:
            cfg_mesh_min = tuple([round(x, 2) for x in self.mesh_config.getfloatlist("mesh_min", count=2)])
            rounded_min = (round(mesh_min[0], 2), round(mesh_min[1], 2))
            if cfg_mesh_min != rounded_min:
                raw["MESH_MIN"] = f"{mesh_min[0]:.2f},{mesh_min[1]:.2f}"
                logger.info("Updating MESH_MIN from config value %s to %s", cfg_mesh_min, rounded_min)

        if mesh_max is not None and "MESH_MAX" not in raw:
            cfg_mesh_max = tuple(round(x, 2) for x in self.mesh_config.getfloatlist("mesh_max", count=2))
            rounded_max = (round(mesh_max[0], 2), round(mesh_max[1], 2))
            if cfg_mesh_max != rounded_max:
                raw["MESH_MAX"] = f"{mesh_max[0]:.2f},{mesh_max[1]:.2f}"
                logger.info("Updating MESH_MAX from config value %s to %s", cfg_mesh_max, rounded_max)

        self._prepare(params)

        points = self._bed_mesh.bmc.probe_mgr.get_std_path()
        if len(points) == 0:
            msg = "probe points are not set"
            raise RuntimeError(msg)
        return [MeshPoint(p[0], p[1], True) for p in points]

    @override
    def finalize(self, offset: Position, positions: list[Position]):
        self._bed_mesh.bmc.probe_finalize(
            [offset.x, offset.y, offset.z],
            [[p.x, p.y, p.z] for p in positions],
        )
