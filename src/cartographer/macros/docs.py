"""Generate macro parameter reference documentation from param() fields.

Usage:
    python -m cartographer.macros.docs > macro-reference.md
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

from cartographer.macros.axis_twist_compensation import AxisTwistCompensationMacro, AxisTwistParams
from cartographer.macros.backlash import EstimateBacklashMacro, EstimateBacklashParams
from cartographer.macros.bed_mesh.scan_mesh import BedMeshCalibrateMacro, BedMeshScanAllParams
from cartographer.macros.fields import ParamInfo, get_all_params
from cartographer.macros.model_manager import ModelManagerParams, ScanModelManager, TouchModelManager
from cartographer.macros.probe import (
    ProbeAccuracyMacro,
    ProbeAccuracyParams,
    ProbeMacro,
    ProbeMacroParams,
    QueryProbeMacro,
    QueryProbeMacroParams,
    ZOffsetApplyProbeMacro,
    ZOffsetApplyProbeMacroParams,
)
from cartographer.macros.query import QueryMacro, QueryParams
from cartographer.macros.scan import ScanAccuracyMacro, ScanAccuracyParams
from cartographer.macros.scan_calibrate import ScanCalibrateMacro, ScanCalibrateParams
from cartographer.macros.stream import StreamMacro, StreamParams
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro, TemperatureCalibrateParams
from cartographer.macros.touch.accuracy import TouchAccuracyMacro, TouchAccuracyParams
from cartographer.macros.touch.calibrate import TouchCalibrateMacro, TouchCalibrateParams
from cartographer.macros.touch.home import TouchHomeMacro, TouchHomeParams
from cartographer.macros.touch.probe import TouchProbeMacro, TouchProbeMacroParams

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Macro

# Macros in the order they should appear in docs.
# (macro_name, macro_class, params_dataclass)
MACROS: list[tuple[str, type[Macro], type]] = [
    # Standard probe macros (no CARTOGRAPHER_ prefix)
    ("PROBE", ProbeMacro, ProbeMacroParams),
    ("PROBE_ACCURACY", ProbeAccuracyMacro, ProbeAccuracyParams),
    ("QUERY_PROBE", QueryProbeMacro, QueryProbeMacroParams),
    ("Z_OFFSET_APPLY_PROBE", ZOffsetApplyProbeMacro, ZOffsetApplyProbeMacroParams),
    ("BED_MESH_CALIBRATE", BedMeshCalibrateMacro, BedMeshScanAllParams),
    # Cartographer macros
    ("CARTOGRAPHER_QUERY", QueryMacro, QueryParams),
    ("CARTOGRAPHER_STREAM", StreamMacro, StreamParams),
    ("CARTOGRAPHER_TEMPERATURE_CALIBRATE", TemperatureCalibrateMacro, TemperatureCalibrateParams),
    ("CARTOGRAPHER_SCAN_CALIBRATE", ScanCalibrateMacro, ScanCalibrateParams),
    ("CARTOGRAPHER_SCAN_ACCURACY", ScanAccuracyMacro, ScanAccuracyParams),
    ("CARTOGRAPHER_SCAN_MODEL", ScanModelManager, ModelManagerParams),
    ("CARTOGRAPHER_ESTIMATE_BACKLASH", EstimateBacklashMacro, EstimateBacklashParams),
    ("CARTOGRAPHER_TOUCH_CALIBRATE", TouchCalibrateMacro, TouchCalibrateParams),
    ("CARTOGRAPHER_TOUCH_MODEL", TouchModelManager, ModelManagerParams),
    ("CARTOGRAPHER_TOUCH_PROBE", TouchProbeMacro, TouchProbeMacroParams),
    ("CARTOGRAPHER_TOUCH_ACCURACY", TouchAccuracyMacro, TouchAccuracyParams),
    ("CARTOGRAPHER_TOUCH_HOME", TouchHomeMacro, TouchHomeParams),
    ("CARTOGRAPHER_AXIS_TWIST_COMPENSATION", AxisTwistCompensationMacro, AxisTwistParams),
]


def _format_default(value: object) -> str:
    """Format a default value for display in docs."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Enum):
        return f"'{value.value}'"
    if isinstance(value, str):
        return f"'{value}'"
    if value is None:
        return "None"
    return str(value)


def _format_param(p: ParamInfo) -> str:
    """Format a single parameter as a documentation line."""
    parts: list[str] = []

    # Type and requirement
    if p.required:
        type_info = f"{p.type}, required"
    elif p.has_default:
        type_info = f"{p.type}, default: {_format_default(p.default)}"
    else:
        type_info = p.type

    parts.append(f"  {p.name} ({type_info})")

    # Description
    if p.description:
        parts[0] += f": {p.description}"

    # Constraints
    constraints: list[str] = []
    if p.min is not None:
        constraints.append(f"minimum: {p.min}")
    if p.max is not None:
        constraints.append(f"maximum: {p.max}")
    if constraints:
        parts.append(f"    Constraints: {', '.join(constraints)}")

    # Allowed values (Enum choices)
    if p.choices:
        parts.append(f"    Allowed values: {', '.join(p.choices)}")

    return "\n".join(parts)


def _format_example_value(p: ParamInfo) -> str:
    """Format a parameter value for the example line."""
    if p.required:
        return f"<{p.name.lower()}>"
    if isinstance(p.default, bool):
        return "yes" if p.default else "no"
    if isinstance(p.default, Enum):
        return str(p.default.value)
    if p.default is None:
        return ""
    return str(p.default)


def generate_docs() -> str:
    """Generate full macro parameter reference as a markdown string."""
    parts: list[str] = []

    parts.append("---")
    parts.append("description: Auto-generated from plugin source.")
    parts.append("---\n")
    parts.append("# Macro Reference\n")

    for macro_name, macro_class, cls in MACROS:
        params = get_all_params(cls)

        parts.append(f"## {macro_name}\n")
        parts.append(f"{macro_class.description}\n")

        if not params:
            parts.append("*This macro accepts no parameters.*\n")
            continue

        parts.append("```")
        for p in params:
            parts.append(_format_param(p))
        parts.append("```\n")

        # Example usage — include required params and optional params with meaningful defaults
        example_parts: list[str] = []
        for p in params:
            value = _format_example_value(p)
            if value:
                example_parts.append(f"{p.name}={value}")
        example = f"{macro_name} {' '.join(example_parts)}".rstrip()
        parts.append("**Example:**")
        parts.append(f"```\n{example}\n```\n")

    return "\n".join(parts)


def main() -> None:
    _ = sys.stdout.write(generate_docs())


if __name__ == "__main__":
    main()
