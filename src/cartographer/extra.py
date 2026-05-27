from __future__ import annotations

import logging

from cartographer import __version__
from cartographer.core import PrinterCartographer
from cartographer.runtime.loader import init_runtime

logger = logging.getLogger(__name__)


def load_config(config: object) -> object:
    adapters, integrator = init_runtime(config)

    integrator.setup()

    cartographer = PrinterCartographer(adapters)

    if cartographer.config.general.register_as_probe:
        integrator.register_probe(cartographer)

    for macro in cartographer.macros:
        integrator.register_macro(macro)

    integrator.register_coil_temperature_sensor()

    chip_name = cartographer.config.general.endstop_chip_name
    integrator.register_endstop_pin(chip_name, "z_virtual_endstop", cartographer.scan_mode)

    integrator.register_ready_callback(cartographer.ready_callback)

    register_reconnect = getattr(adapters.mcu, "register_reconnect_callback", None)
    if register_reconnect is not None:
        register_reconnect(cartographer.validate_and_load_models)

    adapter_name = adapters.__class__.__name__
    logger.info("Loaded Cartographer3D Plugin version %s using %s", __version__, adapter_name)

    return cartographer
