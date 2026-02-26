"""Generate Klipper-style configuration reference documentation from option() fields.

Usage:
    python -m cartographer.config.docs > configuration-reference.md
"""

from __future__ import annotations

import sys
from enum import Enum

from cartographer.config.fields import OptionInfo, get_all_options
from cartographer.interfaces.configuration import (
    BedMeshConfig,
    CoilConfiguration,
    GeneralConfig,
    ScanConfig,
    ScanModelConfiguration,
    TouchConfig,
    TouchModelConfiguration,
)

# Config sections in the order they should appear in docs.
# (section_header, klipper_section_name, dataclass)
SECTIONS: list[tuple[str, str, type]] = [
    ("General", "cartographer", GeneralConfig),
    ("Scan", "cartographer scan", ScanConfig),
    ("Touch", "cartographer touch", TouchConfig),
    ("Bed Mesh", "bed_mesh", BedMeshConfig),
    ("Coil", "cartographer coil", CoilConfiguration),
    ("Scan Model", "cartographer scan_model <name>", ScanModelConfiguration),
    ("Touch Model", "cartographer touch_model <name>", TouchModelConfiguration),
]


def _format_default(value: object) -> str:
    """Format a default value for display in docs."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Enum):
        return f"'{value.value}'"
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _format_option(opt: OptionInfo) -> str:
    """Format a single option as a Klipper-style config reference block."""
    lines: list[str] = []

    # Description
    if opt.description:
        lines.append(f"#   {opt.description}")

    # Constraints
    constraints: list[str] = []
    if opt.min is not None:
        constraints.append(f"minimum: {opt.min}")
    if opt.max is not None:
        constraints.append(f"maximum: {opt.max}")
    if constraints:
        lines.append(f"#   Constraints: {', '.join(constraints)}")

    # Allowed values (Enum choices)
    if opt.choices:
        lines.append(f"#   Allowed values: {', '.join(opt.choices)}")

    # The option line itself
    if opt.required:
        lines.append(f"{opt.name}:")
    elif opt.has_default:
        lines.append(f"#{opt.name}: {_format_default(opt.default)}")
    else:
        lines.append(f"#{opt.name}:")

    return "\n".join(lines)


def generate_docs() -> str:
    """Generate full configuration reference as a markdown string."""
    parts: list[str] = []

    parts.append("---")
    parts.append("description: Auto-generated from plugin source.")
    parts.append("---\n")
    parts.append("# Configuration Reference\n")

    for section_header, section_name, cls in SECTIONS:
        options = get_all_options(cls)
        if not options:
            continue

        parts.append(f"## {section_header}\n")
        parts.append(f"```ini\n[{section_name}]")

        for opt in options:
            parts.append(_format_option(opt))

        parts.append("```\n")

    return "\n".join(parts)


def main() -> None:
    _ = sys.stdout.write(generate_docs())


if __name__ == "__main__":
    main()
