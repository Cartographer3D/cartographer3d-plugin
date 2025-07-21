from __future__ import annotations

from typing import final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams


@final
class MigrationMessageMacro(Macro):
    description = "Implements an old macro and informs of the new"

    def __init__(self, name: str, new_macro: str) -> None:
        self.name = name
        self.new_macro = new_macro

    @override
    def run(self, params: MacroParams) -> None:
        msg = f"Macro {self.name} is deprecated and replaced by {self.new_macro}. Please update your configuration."
        raise RuntimeError(msg)
