from __future__ import annotations

from typing import final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams


@final
class MessageMacro(Macro):
    def __init__(self, message: str) -> None:
        self.description = message

    @override
    def run(self, params: MacroParams) -> None:
        raise RuntimeError(self.description)
