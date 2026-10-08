from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, final

from typing_extensions import override

from cartographer.adapters.kalico.probe import KalicoCartographerProbe
from cartographer.adapters.klipper_like.integrator import (
    KlipperLikeAdapters,
    KlipperLikeIntegrator,
    catch_macro_errors,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from gcode import GCodeCommand
    from klippy import Printer

    from cartographer.core import MacroRegistration, PrinterCartographer


class _ProbeRegistry(Protocol):
    def add_probe_object(self, obj: KalicoCartographerProbe, config: ConfigWrapper) -> object: ...


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
            if callable(getattr(probe_list, "get_list", None)) and callable(
                getattr(probe_list, "add_probe_object", None)
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
