from __future__ import annotations

from contextvars import Context
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from cartographer.adapters.klipper_like.gcode_context import gcode_command_context, get_gcode_command

if TYPE_CHECKING:
    from gcode import GCodeCommand


def command(*methods: str) -> GCodeCommand:
    return cast("GCodeCommand", Mock(spec=["error", *methods], error=ValueError))


def test_nested_scope_restores_on_exception_and_isolates_contexts() -> None:
    outer, inner = command(), command()
    assert get_gcode_command() is None
    with gcode_command_context(outer):
        assert get_gcode_command() is outer
        isolated = Context()
        assert isolated.run(get_gcode_command) is None

        def run_isolated() -> None:
            with gcode_command_context(inner):
                assert get_gcode_command() is inner
            assert get_gcode_command() is None

        isolated.run(run_isolated)
        assert get_gcode_command() is outer
        with pytest.raises(RuntimeError, match="nested failure"), gcode_command_context(inner):
            assert get_gcode_command() is inner
            assert isolated.run(get_gcode_command) is None
            msg = "nested failure"
            raise RuntimeError(msg)
        assert get_gcode_command() is outer
    assert get_gcode_command() is None
