from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, final

from typing_extensions import override

from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper_like.integrator import (
    FallbackMacroAdapter,
    KlipperLikeAdapters,
    KlipperLikeIntegrator,
    catch_macro_errors,
)
from cartographer.interfaces.printer import SupportsFallbackMacro

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand
    from klippy import Printer

    from cartographer.core import MacroRegistration, PrinterCartographer


class _ProbeRegistry(Protocol):
    def add_probe_object(self, obj: KalicoCartographerProbe, config: ConfigWrapper) -> object: ...
    def get_command_probe(self, gcmd: GCodeCommand, default: object = ...) -> object | None: ...


class _ProbeList(Protocol):
    def get_list(self, printer: Printer) -> _ProbeRegistry: ...


@final
class KalicoIntegrator(KlipperLikeIntegrator):
    def __init__(self, adapters: KlipperLikeAdapters) -> None:
        super().__init__(adapters, KalicoCartographerProbe)
        self._probe_list: _ProbeList | None = None
        self._registry_probe: KalicoCartographerProbe | None = None
        try:
            probe_module = import_module("extras.probe")
        except ModuleNotFoundError as error:
            if error.name not in ("extras", "extras.probe"):
                raise
        else:
            probe_list = getattr(probe_module, "ProbeList", None)
            if all(
                callable(getattr(probe_list, method, None))
                for method in ("get_list", "add_probe_object", "get_command_probe")
            ):
                self._probe_list = cast("_ProbeList", probe_list)

    @override
    def register_probe(self, cartographer: PrinterCartographer) -> None:
        if self._probe_list is None:
            super().register_probe(cartographer)
            return

        probe = KalicoCartographerProbe(
            self._toolhead,
            cartographer.scan_mode,
            cartographer.probe_macro,
            cartographer.query_probe_macro,
            cartographer.config.general,
            printer=self._printer,
        )
        registry = self._probe_list.get_list(self._printer)
        _ = registry.add_probe_object(probe, self._config.wrapper)
        self._registry_probe = probe
        if not probe.is_default_probe:
            for registration in cartographer.probe_macros:
                self.register_macro(registration)

    @override
    def register_macro(self, registration: MacroRegistration) -> None:
        probe = self._registry_probe
        if probe is not None and registration.name == "BED_MESH_CALIBRATE":
            assert self._probe_list is not None
            registry = self._probe_list.get_list(self._printer)
            macro = registration.macro
            original = self._gcode.register_command(registration.name, None)
            if original is not None and isinstance(macro, SupportsFallbackMacro):
                macro.set_fallback_macro(FallbackMacroAdapter(registration.name, original))

            def route_mesh(gcmd: GCodeCommand) -> None:
                selected = registry.get_command_probe(gcmd, None)
                method = gcmd.get("METHOD", None)
                if method is not None:
                    method = method.lower()
                if selected is None and method != "manual":
                    # Ask the registry for its native missing-default error before any work.
                    selected = registry.get_command_probe(gcmd)
                if selected is probe and method in (None, "scan"):
                    params = {key: value for key, value in gcmd.get_command_parameters().items() if key != "PROBE"}
                    command = self._gcode.create_gcode_command(gcmd.get_command(), gcmd.get_commandline(), params)
                    macro.run(command)
                    return
                if method == "scan":
                    msg = "METHOD=scan requires the Cartographer probe"
                    raise gcmd.error(msg)
                if original is None:
                    msg = "The native BED_MESH_CALIBRATE handler is unavailable"
                    raise gcmd.error(msg)
                if selected is probe and method != "manual":
                    macro.run(gcmd)
                else:
                    original(gcmd)

            self._gcode.register_command(registration.name, catch_macro_errors(route_mesh), desc=macro.description)
            return

        if probe is None or registration.name not in (
            "PROBE",
            "PROBE_ACCURACY",
            "QUERY_PROBE",
            "Z_OFFSET_APPLY_PROBE",
        ):
            super().register_macro(registration)
            return

        def run(gcmd: GCodeCommand) -> None:
            params = {key: value for key, value in gcmd.get_command_parameters().items() if key != "PROBE"}
            command = self._gcode.create_gcode_command(gcmd.get_command(), gcmd.get_commandline(), params)
            registration.macro.run(command)

        handler = catch_macro_errors(run)
        self._gcode.register_mux_command(
            registration.name,
            "PROBE",
            probe.probe_name,
            handler,
            desc=registration.macro.description,
        )
        if probe.is_default_probe:
            self._gcode.register_mux_command(
                registration.name,
                "PROBE",
                None,
                handler,
                desc=registration.macro.description,
            )
