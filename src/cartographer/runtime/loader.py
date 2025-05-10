from typing import TYPE_CHECKING, cast

from cartographer.adapters.klipper.adapters import KlipperAdapters
from cartographer.adapters.klipper.integrator import KlipperIntegrator
from cartographer.runtime.adapters import Adapters
from cartographer.runtime.environment import Environment, detect_environment
from cartographer.runtime.integrator import Integrator

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper


def init_adapter(config: object) -> Adapters:
    env = detect_environment(config)
    if env == Environment.Klipper:
        from cartographer.adapters.klipper.adapters import KlipperAdapters

        return KlipperAdapters(cast("KlipperConfigWrapper", config))

    msg = f"unsupported environment: {env}"
    raise RuntimeError(msg)


def init_integrator(adapters: Adapters) -> Integrator:
    if isinstance(adapters, KlipperAdapters):
        return KlipperIntegrator(adapters)

    msg = "Unsupported adapters"
    raise RuntimeError(msg)
