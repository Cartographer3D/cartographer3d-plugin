from __future__ import annotations

import logging

from cartographer import __version__
from cartographer.core import PrinterCartographer
from cartographer.runtime.loader import init_adapter, init_integrator

logger = logging.getLogger(__name__)


def load_config(config: object) -> object:
    adapters = init_adapter(config)
    integrator = init_integrator(adapters)

    integrator.setup()

    cartographer = PrinterCartographer(adapters)

    integrator.register_cartographer(cartographer)

    for macro in cartographer.macros:
        integrator.register_macro(macro)

    integrator.register_coil_temperature_sensor()
    integrator.register_endstop_pin("probe", "z_virtual_endstop", cartographer.scan_mode)

    integrator.register_ready_callback(cartographer.ready_callback)

    integrator_name = integrator.__class__.__name__
    logger.info("Loaded Cartographer3D Plugin version %s using %s", __version__, integrator_name)

    return cartographer
