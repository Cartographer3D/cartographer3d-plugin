from __future__ import annotations

import dataclasses
import logging
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from cartographer import __version__

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from cartographer.runtime.adapters import Adapters

logger = logging.getLogger(__name__)

# Fields excluded from per-section serialisation.
_GENERAL_SKIP: set[str] = {"mcu"}
_SCAN_SKIP: set[str] = {"models"}
_TOUCH_SKIP: set[str] = {"models"}
# faulty_regions is a nested tuple structure and not useful for high-level analytics.
_BED_MESH_SKIP: set[str] = {"faulty_regions"}


def _to_json_value(val: object) -> object:
    """Recursively convert *val* to a JSON-serialisable type.

    * ``Enum`` → ``.value``
    * ``Sequence`` (tuple, list) → ``list`` (recursive)
    * Everything else is returned as-is (caller's responsibility to ensure
      JSON-compatibility for complex types).
    """
    if isinstance(val, Enum):
        return _to_json_value(val.value)
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        return [_to_json_value(item) for item in val]
    return val


def _dataclass_to_dict(obj: DataclassInstance, skip: set[str] | None = None) -> dict[str, object]:
    """Serialise a dataclass instance to a JSON-compatible dict.

    Fields listed in *skip* are omitted.  ``ClassVar`` fields are
    automatically excluded by ``dataclasses.fields()``.
    """
    result: dict[str, object] = {}
    for f in dataclasses.fields(obj):
        if skip and f.name in skip:
            continue
        result[f.name] = _to_json_value(getattr(obj, f.name))
    return result


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def collect_startup_data(adapters: Adapters, anonymous_id: str) -> dict[str, object]:
    """Build the startup telemetry payload."""
    cfg = adapters.config

    return {
        "timestamp": _utc_now(),
        "event": "startup",
        "anonymous_id": anonymous_id,
        "plugin_version": __version__,
        "environment": adapters.environment,
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "os_arch": platform.machine(),
        "mcu_version": adapters.mcu.get_mcu_version(),
        "config": {
            "general": _dataclass_to_dict(cfg.general, skip=_GENERAL_SKIP),
            "scan": _dataclass_to_dict(cfg.scan, skip=_SCAN_SKIP),
            "touch": _dataclass_to_dict(cfg.touch, skip=_TOUCH_SKIP),
            "bed_mesh": _dataclass_to_dict(cfg.bed_mesh, skip=_BED_MESH_SKIP),
            "coil": {"has_calibration": cfg.coil.calibration is not None},
        },
        # TODO: expose kinematics type via the Toolhead protocol.
        "kinematics": "unknown",
        "bed_size": {
            "min": list(cfg.bed_mesh.mesh_min),
            "max": list(cfg.bed_mesh.mesh_max),
        },
        "calibration_summary": {
            "scan_models_count": len(cfg.scan.models),
            "touch_models_count": len(cfg.touch.models),
            "has_temperature_calibration": cfg.coil.calibration is not None,
        },
    }
