from __future__ import annotations

from unittest.mock import Mock

import pytest

import cartographer.adapters.klipper_like.utils as _utils_module
from cartographer.adapters.klipper_like.integrator import catch_macro_errors
from cartographer.adapters.klipper_like.utils import reraise_for_klipper
from cartographer.interfaces.errors import PrinterShutdownError


class _FakeCommandError(Exception):
    """Stand-in for gcode.CommandError used throughout this module."""


class TestPrinterShutdownErrorPropagation:
    # ------------------------------------------------------------------ #
    # reraise_for_klipper                                                  #
    # ------------------------------------------------------------------ #

    def test_reraise_for_klipper_converts_printer_shutdown_to_command_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PrinterShutdownError becomes CommandError with the shutdown message."""
        monkeypatch.setattr(_utils_module, "CommandError", _FakeCommandError)

        @reraise_for_klipper
        def func() -> None:
            raise PrinterShutdownError()

        with pytest.raises(_FakeCommandError, match="Aborted: printer entered shutdown"):
            func()

    def test_reraise_for_klipper_converts_runtime_error_to_command_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError is re-raised as CommandError, preserving the message."""
        monkeypatch.setattr(_utils_module, "CommandError", _FakeCommandError)
        msg = "some msg"

        @reraise_for_klipper
        def func() -> None:
            raise RuntimeError(msg)

        with pytest.raises(_FakeCommandError, match="some msg"):
            func()

    def test_reraise_for_klipper_passes_through_non_runtime_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exceptions that are not RuntimeError subclasses pass through unchanged."""
        monkeypatch.setattr(_utils_module, "CommandError", _FakeCommandError)

        @reraise_for_klipper
        def func() -> None:
            raise KeyError()

        with pytest.raises(KeyError):
            func()

    # ------------------------------------------------------------------ #
    # catch_macro_errors                                                  #
    # ------------------------------------------------------------------ #

    def test_catch_macro_errors_converts_printer_shutdown(self) -> None:
        """PrinterShutdownError inside a macro handler becomes gcmd.error(shutdown msg)."""
        gcmd = Mock()
        gcmd.error = _FakeCommandError

        @catch_macro_errors
        def handler(_gcmd: object) -> None:
            raise PrinterShutdownError()

        with pytest.raises(_FakeCommandError, match="Aborted: printer entered shutdown"):
            handler(gcmd)

    def test_catch_macro_errors_converts_runtime_error(self) -> None:
        """RuntimeError inside a macro handler becomes gcmd.error(msg)."""
        gcmd = Mock()
        gcmd.error = _FakeCommandError

        msg = "runtime msg"

        @catch_macro_errors
        def handler(_gcmd: object) -> None:
            raise RuntimeError(msg)

        with pytest.raises(_FakeCommandError, match="runtime msg"):
            handler(gcmd)

    def test_catch_macro_errors_converts_value_error(self) -> None:
        """ValueError inside a macro handler becomes gcmd.error(msg)."""
        gcmd = Mock()
        gcmd.error = _FakeCommandError

        msg = "value msg"

        @catch_macro_errors
        def handler(_gcmd: object) -> None:
            raise ValueError(msg)

        with pytest.raises(_FakeCommandError, match="value msg"):
            handler(gcmd)
