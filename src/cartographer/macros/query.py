from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.utils import get_enum_choice

if TYPE_CHECKING:
    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode

logger = logging.getLogger(__name__)


class QueryField(Enum):
    TOUCH = "touch"
    SCAN = "scan"
    MCU = "mcu"
    ALL = "all"


@final
class QueryMacro(Macro):
    description: str = "Query current cartographer state"

    def __init__(self, mcu: Mcu, scan: ScanMode, touch: TouchMode) -> None:
        self._mcu = mcu
        self._scan = scan
        self._touch = touch

    @override
    def run(self, params: MacroParams) -> None:
        field = get_enum_choice(params, "FIELD", QueryField, default=QueryField.ALL)
        time = self._mcu.get_current_time()
        if field == QueryField.MCU or field == QueryField.ALL:
            logger.info(_format_status(self._mcu.get_status(time), "Cartographer Mcu Status"))
        if field == QueryField.SCAN or field == QueryField.ALL:
            logger.info(_format_status(self._scan.get_status(time), "Cartographer Scan Mode Status"))
        if field == QueryField.TOUCH or field == QueryField.ALL:
            logger.info(_format_status(self._touch.get_status(time), "Cartographer Touch Mode Status"))


def _format_status(status_dict: dict[str, object], title: str | None = None, indent: str = "  ") -> str:
    lines: list[str] = []

    if title:
        lines.append(f"{title}:")

    for key, value in status_dict.items():
        formatted_key = _format_key(key)
        formatted_value = _format_value(value)
        lines.append(f"{indent}{formatted_key}: {formatted_value}")

    return "\n".join(lines)


def _format_key(key: str) -> str:
    """Convert snake_case keys to readable labels."""
    # Convert snake_case to words
    words = key.replace("_", " ").split()
    # Capitalize each word
    return " ".join(word.capitalize() for word in words)


def _format_value(value: object) -> str:
    """Format values with appropriate units and formatting."""
    if value is None:
        return "None"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        # Check if this looks like a coordinate or measurement
        if isinstance(value, float):
            # Format floats with appropriate precision
            if abs(value) < 0.001:
                return f"{value:.6f}"
            elif abs(value) < 1:
                return f"{value:.4f}"
            else:
                return f"{value:.3f}"
        return str(value)

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        # Handle nested dictionaries
        nested_lines: list[str] = []
        for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
            formatted_k = _format_key(k)  # pyright: ignore[reportUnknownArgumentType]
            formatted_v = _format_value(v)  # pyright: ignore[reportUnknownArgumentType]
            nested_lines.append(f"    {formatted_k}: {formatted_v}")
        return "\n" + "\n".join(nested_lines)

    if isinstance(value, (list, tuple)):
        if not value:
            return "None"
        return ", ".join(map(_format_value, value))  # pyright: ignore[reportUnknownArgumentType]

    return str(value)
