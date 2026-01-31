from __future__ import annotations

from typing import overload

from typing_extensions import override

from cartographer.interfaces.printer import MacroParams


class MockParams(MacroParams):
    def __init__(self) -> None:
        self.params: dict[str, str] = {}

    @override
    def get_command_parameters(self) -> dict[str, str]:
        return self.params

    @overload
    def get(self, name: str, default: str = ...) -> str: ...
    @overload
    def get(self, name: str, default: None) -> str | None: ...

    @override
    def get(self, name: str, default: str | None = ...) -> str | None:
        opt = self.params.get(name, default)
        return str(opt) if opt is not None else None

    @overload
    def get_float(
        self, name: str, default: float = ..., *, above: float = ..., minval: float = ..., maxval: float = ...
    ) -> float: ...
    @overload
    def get_float(
        self, name: str, default: None, *, above: float = ..., minval: float = ..., maxval: float = ...
    ) -> float | None: ...

    @override
    def get_float(
        self, name: str, default: float | None = ..., *, above: float = ..., minval: float = ..., maxval: float = ...
    ) -> float | None:
        opt = self.params.get(name, default)
        return float(opt) if opt is not None else None

    @override
    def get_int(self, name: str, default: int = ..., *, minval: int = ..., maxval: int = ...) -> int:
        return int(self.params.get(name, default))
