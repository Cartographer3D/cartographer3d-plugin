from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator
from cartographer.adapters.klipper_v12.probe import KlipperV12CartographerProbe
from cartographer.mcu.mcu import CartographerMcu

if TYPE_CHECKING:
    from cartographer.adapters.klipper_v12.adapters import KlipperV12Adapters
    from cartographer.core import PrinterCartographer

logger = logging.getLogger(__name__)


@final
class KlipperV12Integrator(KlipperLikeIntegrator):
    def __init__(self, adapters: KlipperV12Adapters) -> None:
        assert isinstance(adapters.mcu, CartographerMcu), "Invalid MCU type for KlipperV12Integrator"
        super().__init__(adapters)
        self._toolhead = adapters.toolhead

    @override
    def register_cartographer(self, cartographer: PrinterCartographer) -> None:
        self._printer.add_object(
            "probe",
            KlipperV12CartographerProbe(
                self._toolhead,
                cartographer.scan_mode,
                cartographer.probe_macro,
                cartographer.query_probe_macro,
                cartographer.config.general,
            ),
        )
