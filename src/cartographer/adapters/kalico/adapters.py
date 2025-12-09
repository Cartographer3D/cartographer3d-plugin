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
from cartographer.runtime.adapters import Adapters

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper


logger = logging.getLogger(__name__)


@final
class KalicoAdapters(Adapters):
    def __init__(self, config: KlipperConfigWrapper) -> None:
        self.printer = config.get_printer()
        self.scheduler = KlipperScheduler(self.printer.get_reactor())

        self.mcu = KlipperCartographerMcu(config, self.scheduler)
        self.config = KlipperConfiguration(config, self.mcu)

        self.toolhead = KalicoToolhead(config, self.mcu)
        self.bed_mesh = KlipperBedMesh(config)
        self.gcode = KlipperGCodeDispatch(self.printer)

        self.axis_twist_compensation = None
        if config.has_section("axis_twist_compensation"):
            self.axis_twist_compensation = KalicoAxisTwistCompensationAdapter(config)
