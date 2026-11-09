from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import Scheduler
    from cartographer.interfaces.printer import AxisTwistCompensation, GCodeDispatch, Mcu, Toolhead
    from cartographer.macros.axis_twist_compensation import AxisTwistCompensationAdapter
    from cartographer.macros.bed_mesh.interfaces import BedMeshAdapter


class Adapters(Protocol):
    config: Configuration
    toolhead: Toolhead
    mcu: Mcu
    axis_twist_compensation: AxisTwistCompensation | None
    axis_twist_calibration: AxisTwistCompensationAdapter | None
    bed_mesh: BedMeshAdapter
    scheduler: Scheduler
    gcode: GCodeDispatch
