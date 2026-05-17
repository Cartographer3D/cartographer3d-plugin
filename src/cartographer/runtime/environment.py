from __future__ import annotations

from enum import Enum


class Environment(Enum):
    Klipper2024 = "klipper_2024"
    Klipper = "klipper"
    Kalico = "kalico"
    KlipperV12 = "klipper_v12"


def detect_environment(config: object) -> Environment:
    del config
    try:
        from klippy import APP_NAME

        if APP_NAME == "Kalico":
            return Environment.Kalico
    except ImportError:
        pass

    try:
        from mcu import TriggerDispatch

        del TriggerDispatch
    except ImportError:
        return Environment.KlipperV12

    # TODO: Differentiate 2024 and main
    return Environment.Klipper
