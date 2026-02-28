from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.adapters.kalico.axis_twist_compensation import KalicoAxisTwistCompensationAdapter
from cartographer.adapters.kalico.toolhead import KalicoToolhead
from cartographer.adapters.klipper.bed_mesh import KlipperBedMesh
from cartographer.adapters.klipper.configuration import KlipperConfiguration
from cartographer.adapters.klipper.gcode import KlipperGCodeDispatch
from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
from cartographer.adapters.klipper.scheduler import KlipperScheduler
from cartographer.adapters.klipper_like.utils import try_load_object
from cartographer.config.fields import parse
from cartographer.interfaces.configuration import GeneralConfig
from cartographer.runtime.adapters import Adapters

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper


logger = logging.getLogger(__name__)


@final
class KalicoAdapters(Adapters):
    def __init__(self, config: KlipperConfigWrapper) -> None:
        self.printer = config.get_printer()
        self.scheduler = KlipperScheduler(self.printer.get_reactor())

        general = parse(GeneralConfig, config)
        self.mcu = KlipperCartographerMcu(config, self.scheduler, general.mcu)
        self.config = KlipperConfiguration(config, self.mcu, general)

        self.toolhead = KalicoToolhead(config, self.mcu)
        self.bed_mesh = KlipperBedMesh(config)
        self.gcode = KlipperGCodeDispatch(self.printer)

        self.axis_twist_compensation = None
        if config.has_section("axis_twist_compensation"):
            self.axis_twist_compensation = KalicoAxisTwistCompensationAdapter(config)

        self.probe_method_macros = ["BED_MESH_CALIBRATE"]
        if try_load_object(self.printer, config, "z_tilt") or try_load_object(self.printer, config, "z_tilt_ng"):
            self.probe_method_macros.append("Z_TILT_ADJUST")
        if try_load_object(self.printer, config, "quad_gantry_level"):
            self.probe_method_macros.append("QUAD_GANTRY_LEVEL")
        if try_load_object(self.printer, config, "screws_tilt_adjust"):
            self.probe_method_macros.append("SCREWS_TILT_CALCULATE")
