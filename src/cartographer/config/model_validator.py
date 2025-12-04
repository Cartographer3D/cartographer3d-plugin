"""Model validation at startup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

MINIMUM_SCAN_MODEL_VERSION = "1.1.0"
MINIMUM_TOUCH_MODEL_VERSION = "1.1.0"

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import (
        Configuration,
        ModelVersionInfo,
    )

logger = logging.getLogger(__name__)


def compare_versions(version_a: str, version_b: str) -> int:
    """
    Compare two version strings.

    Returns -1 if a < b, 0 if a == b, 1 if a > b.
    """

    def parse(v: str) -> tuple[int, ...]:
        parts: list[int] = []
        for part in v.split("+")[0].split("-")[0].split("."):
            try:
                parts.append(int(part))
            except ValueError:
                break
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    pa, pb = parse(version_a), parse(version_b)
    return -1 if pa < pb else (1 if pa > pb else 0)


def _is_model_compatible(
    version_info: ModelVersionInfo | None,
    current_mcu_version: str | None,
    min_software_version: str,
) -> tuple[bool, str | None]:
    """
    Check if a model is compatible with current versions.

    Returns (is_compatible, reason_if_not).
    """
    if version_info is None:
        return False, "created before version tracking"

    if (
        version_info.mcu_version is not None
        and current_mcu_version is not None
        and version_info.mcu_version != current_mcu_version
    ):
        return False, (f"MCU version mismatch (model: {version_info.mcu_version}, current: {current_mcu_version})")

    if (
        version_info.software_version is not None
        and compare_versions(version_info.software_version, min_software_version) < 0
    ):
        return False, (f"software too old (model: {version_info.software_version}, minimum: {min_software_version})")

    return True, None


def validate_and_remove_incompatible_models(
    config: Configuration,
    mcu_version: str | None,
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
            MINIMUM_SCAN_MODEL_VERSION,
        )
        if not compatible:
            logger.warning(
                "Removing incompatible scan model '%s':\n%s.\nPlease recalibrate.",
                name,
                reason,
            )
            config.remove_scan_model(name)
        elif reason:
            logger.warning("Scan model '%s': %s\nConsider recalibrating.", name, reason)

    # Validate touch models
    for name in list(config.touch.models.keys()):
        model = config.touch.models[name]
        compatible, reason = _is_model_compatible(
            model.version_info,
            mcu_version,
            MINIMUM_TOUCH_MODEL_VERSION,
        )
        if not compatible:
            logger.warning(
                "Removing incompatible touch model '%s': %s. Please recalibrate.",
                name,
                reason,
            )
            config.remove_touch_model(name)
        elif reason:
            logger.warning("Touch model '%s': %s\nConsider recalibrating.", name, reason)
