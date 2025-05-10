from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from cartographer.adapters.shared.axis_twist_compensation import KlipperAxisTwistCompensationHelper
from cartographer.adapters.shared.configuration import KlipperConfiguration
from cartographer.adapters.shared.mcu.mcu import KlipperCartographerMcu
from cartographer.adapters.shared.printer import KlipperToolhead
from cartographer.runtime.types import Adapters

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper


logger = logging.getLogger(__name__)


@final
class KlipperAdapters(Adapters):
    def __init__(self, config: KlipperConfigWrapper) -> None:
        self.printer = config.get_printer()

        self.config = KlipperConfiguration(config)
        self.mcu = KlipperCartographerMcu(config)
        self.toolhead = KlipperToolhead(config, self.mcu)

        if config.has_section("axis_twist_compensation"):
            self.axis_twist_compensation = KlipperAxisTwistCompensationHelper(config)
