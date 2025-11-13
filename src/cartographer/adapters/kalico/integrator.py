from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu
from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

if TYPE_CHECKING:
    from cartographer.adapters.kalico.adapters import KalicoAdapters
    from cartographer.core import PrinterCartographer

logger = logging.getLogger(__name__)


@final
class KalicoIntegrator(KlipperLikeIntegrator):
    def __init__(self, adapters: KalicoAdapters) -> None:
        assert isinstance(adapters.mcu, KlipperCartographerMcu), "Invalid MCU type for KalicoIntegrator"
        super().__init__(adapters)
        self._toolhead = adapters.toolhead

    @override
    def register_cartographer(self, cartographer: PrinterCartographer) -> None:
        self._printer.add_object(
            "probe",
            KalicoCartographerProbe(
                self._toolhead,
                cartographer.scan_mode,
                cartographer.probe_macro,
                cartographer.query_probe_macro,
                cartographer.config.general,
            ),
        )
