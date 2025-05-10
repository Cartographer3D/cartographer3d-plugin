from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cartographer.core import PrinterCartographer
    from cartographer.interfaces.printer import Endstop, Macro


class Integrator(Protocol):
    def setup(self) -> None: ...
    def register_cartographer(self, cartographer: PrinterCartographer) -> None: ...
    def register_macro(self, macro: Macro) -> None: ...
    def register_endstop_pin(self, chip_name: str, pin: str, endstop: Endstop) -> None: ...
