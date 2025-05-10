from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import Mcu, Toolhead
    from cartographer.macros.axis_twist_compensation import AxisTwistCompensationHelper


class Adapters(Protocol):
    config: Configuration
    toolhead: Toolhead
    mcu: Mcu
    axis_twist_compensation: AxisTwistCompensationHelper | None
