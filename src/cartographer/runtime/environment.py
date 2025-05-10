from __future__ import annotations

from enum import Enum


class Environment(Enum):
    Klipper2024 = "klipper_2024"
    Klipper = "klipper"
    Kalico = "kalico"


def detect_environment(config: object) -> Environment:
    del config
    try:
        from klippy import APP_NAME

        if APP_NAME == "Kalico":
            return Environment.Kalico
    except ImportError:
        pass

    # TODO: Differentiate 2024 and main
    return Environment.Klipper
