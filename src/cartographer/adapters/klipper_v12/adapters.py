from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.adapters.klipper.bed_mesh import KlipperBedMesh
from cartographer.adapters.klipper.configuration import KlipperConfiguration
from cartographer.adapters.klipper.gcode import KlipperGCodeDispatch
from cartographer.adapters.klipper.scheduler import KlipperScheduler
from cartographer.adapters.klipper.toolhead import KlipperToolhead
from cartographer.adapters.klipper_v12.axis_twist_compensation import KlipperV12AxisTwistCompensationAdapter
from cartographer.adapters.klipper_v12.mcu_platform import KlipperV12McuPlatform
from cartographer.config.fields import parse
from cartographer.interfaces.configuration import GeneralConfig
from cartographer.mcu.mcu import CartographerMcu
from cartographer.runtime.adapters import Adapters

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper


logger = logging.getLogger(__name__)


@final
class KlipperV12Adapters(Adapters):
    def __init__(self, config: KlipperConfigWrapper) -> None:
        self.printer = config.get_printer()
        self.scheduler = KlipperScheduler(self.printer.get_reactor())

        general = parse(GeneralConfig, config)
        platform = KlipperV12McuPlatform(config, general.mcu)
        self.mcu = CartographerMcu(platform, self.scheduler)
        self.config = KlipperConfiguration(config, self.mcu, general)

        self.toolhead = KlipperToolhead(config, self.mcu)
        self.bed_mesh = KlipperBedMesh(config)
        self.gcode = KlipperGCodeDispatch(self.printer)

        self.axis_twist_compensation = None
        if config.has_section("axis_twist_compensation"):
            self.axis_twist_compensation = KlipperV12AxisTwistCompensationAdapter(config)
