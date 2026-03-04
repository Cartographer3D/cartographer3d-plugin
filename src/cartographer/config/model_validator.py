from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

MINIMUM_SCAN_MODEL_VERSION = (1, 0, 0)
MINIMUM_TOUCH_MODEL_VERSION = (1, 1, 0)

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import (
        Configuration,
        ModelVersionInfo,
    )

logger = logging.getLogger(__name__)


def meets_minimum_version(version: str, minimum: tuple[int, int, int]) -> bool:
    """Check if version >= minimum (minimum is always a stable release)."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        return False

    v_base = tuple(int(x or 0) for x in match.groups())

    return v_base >= minimum


def _is_model_compatible(
    version_info: ModelVersionInfo,
    current_mcu_version: str,
    min_software_version: tuple[int, int, int],
) -> tuple[bool, str | None]:
    """
    Check if a model is compatible with current versions.

    Returns (is_compatible, reason_if_not).
    """
    if version_info.mcu_version is not None and version_info.mcu_version != current_mcu_version:
        return False, (f"MCU version mismatch (model: {version_info.mcu_version}, current: {current_mcu_version})")

    if not meets_minimum_version(version_info.software_version, min_software_version):
        return False, (
            "software version too old "
            f"(model: {version_info.software_version}, minimum: {'.'.join(map(str, min_software_version))})"
        )

    if version_info.mcu_version is None:
        return True, "created before version tracking"

    return True, None


def validate_and_remove_incompatible_models(
    config: Configuration,
    mcu_version: str,
    minimum_scan_version: tuple[int, int, int] = MINIMUM_SCAN_MODEL_VERSION,
    minimum_touch_version: tuple[int, int, int] = MINIMUM_TOUCH_MODEL_VERSION,
) -> None:
    """
    Validate all models and remove incompatible ones.
    """

    # Validate scan models
    for name in list(config.scan.models.keys()):
        model = config.scan.models[name]
        compatible, reason = _is_model_compatible(
            model.version_info,
            mcu_version,
            minimum_scan_version,
        )
        if not compatible:
            config.log_runtime_warning(
                f"[cartographer] Removing incompatible scan model '{name}': {reason}. Please recalibrate."
            )
            config.remove_scan_model(name)
        elif reason:
            config.log_runtime_warning(f"[cartographer] Old scan model '{name}': {reason}. Consider recalibrating.")

    # Validate touch models
    for name in list(config.touch.models.keys()):
        model = config.touch.models[name]
        compatible, reason = _is_model_compatible(
            model.version_info,
            mcu_version,
            minimum_touch_version,
        )
        if not compatible:
            config.log_runtime_warning(
                f"[cartographer] Removing incompatible touch model '{name}' {reason}. Please recalibrate."
            )
            config.remove_touch_model(name)
        elif reason:
            config.log_runtime_warning(f"[cartographer] Old touch model '{name}' {reason}. Consider recalibrating.")
