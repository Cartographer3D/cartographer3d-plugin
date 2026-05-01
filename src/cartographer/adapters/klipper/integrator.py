from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper.probe import KlipperCartographerProbe
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator
from cartographer.mcu.mcu import CartographerMcu

if TYPE_CHECKING:
    from cartographer.adapters.klipper.adapters import KlipperAdapters
    from cartographer.core import PrinterCartographer

logger = logging.getLogger(__name__)


@final
class KlipperIntegrator(KlipperLikeIntegrator):
    def __init__(self, adapters: KlipperAdapters) -> None:
        assert isinstance(adapters.mcu, CartographerMcu), "Invalid MCU type for KlipperIntegrator"
        super().__init__(adapters)
        self._toolhead = adapters.toolhead

    @override
    def register_cartographer(self, cartographer: PrinterCartographer) -> None:
        self._printer.add_object(
            "probe",
            KlipperCartographerProbe(
                self._toolhead,
                cartographer.scan_mode,
                cartographer.probe_macro,
                cartographer.query_probe_macro,
                cartographer.config.general,
            ),
        )
