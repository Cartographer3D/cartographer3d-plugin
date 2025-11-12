from __future__ import annotations

from typing import final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams


@final
class MigrationMessageMacro(Macro):
    def __init__(self, name: str, new_macro: str) -> None:
        self.description = f"Macro {name} has been replaced by {new_macro}."

    @override
    def run(self, params: MacroParams) -> None:
        msg = f"{self.description} Please update your configuration."
        raise RuntimeError(msg)
