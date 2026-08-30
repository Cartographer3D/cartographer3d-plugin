from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from cartographer.runtime.environment import Environment, detect_environment

if TYPE_CHECKING:
    from configfile import ConfigWrapper as KlipperConfigWrapper

    from cartographer.runtime.adapters import Adapters
    from cartographer.runtime.integrator import Integrator

logger = logging.getLogger(__name__)


def init_runtime(config: object) -> tuple[Adapters, Integrator]:
    env = detect_environment(config)
    logger.info("Detected environment: %s", env.value)

    if env == Environment.Klipper:
        from cartographer.adapters.klipper.adapters import KlipperAdapters
        from cartographer.adapters.klipper.probe import KlipperCartographerProbe
        from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

        adapters = KlipperAdapters(cast("KlipperConfigWrapper", config))
        return adapters, KlipperLikeIntegrator(adapters, KlipperCartographerProbe)

    if env == Environment.KlipperV12:
        from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator
        from cartographer.adapters.klipper_v12.adapters import KlipperV12Adapters
        from cartographer.adapters.klipper_v12.probe import KlipperV12CartographerProbe

        adapters = KlipperV12Adapters(cast("KlipperConfigWrapper", config))
        return adapters, KlipperLikeIntegrator(adapters, KlipperV12CartographerProbe)

    if env == Environment.Kalico:
        from cartographer.adapters.kalico.adapters import KalicoAdapters
        from cartographer.adapters.kalico.probe import KalicoCartographerProbe
        from cartographer.adapters.klipper_like.integrator import KlipperLikeIntegrator

        adapters = KalicoAdapters(cast("KlipperConfigWrapper", config))
        return adapters, KlipperLikeIntegrator(adapters, KalicoCartographerProbe)

    msg = f"Unsupported environment: {env}"
    raise RuntimeError(msg)
