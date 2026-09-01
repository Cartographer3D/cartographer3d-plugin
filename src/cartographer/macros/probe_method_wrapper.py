from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, SupportsFallbackMacro

if TYPE_CHECKING:
    from cartographer.probe import Probe

_VALID_PROBE_METHODS: frozenset[str] = frozenset({"scan", "touch"})


@final
class ProbeMethodWrapperMacro(SupportsFallbackMacro):
    description = "Select probe method (scan/touch) for leveling commands."
    # Registration must fail if no existing handler is found.
    requires_fallback: bool = True

    def __init__(self, probe: Probe, gcode: GCodeDispatch, command_name: str) -> None:
        self._fallback: Macro | None = None
        self.probe = probe
        self.gcode = gcode
        self.command_name = command_name

    @property
    def fallback(self) -> Macro:
        if self._fallback is None:
            msg = f"Fallback for {type(self).__name__} not found."
            raise RuntimeError(msg)
        return self._fallback

    @override
    def set_fallback_macro(self, macro: Macro) -> None:
        self._fallback = macro

    @override
    def run(self, params: MacroParams) -> None:
        probe_method = params.get("PROBE_METHOD", "scan").lower()

        if probe_method not in _VALID_PROBE_METHODS:
            msg = f"Unknown PROBE_METHOD={probe_method!r}. Valid values: {sorted(_VALID_PROBE_METHODS)!r}"
            raise ValueError(msg)

        if probe_method != "touch":
            # Scan mode (default): delegate original params unchanged.
            self.fallback.run(params)
            return

        # Touch mode.
        if self.command_name == "BED_MESH_CALIBRATE":
            self._run_touch_mesh(params)
        else:
            with self.probe.as_touch():
                self.fallback.run(params)

    def _run_touch_mesh(self, params: MacroParams) -> None:
        """Handle touch mode for mesh-calibration commands (BED_MESH_CALIBRATE).

        - METHOD absent is treated as "automatic".
        - Explicit METHOD=automatic is accepted.
        - Any other explicit METHOD raises before movement.
        - Supplies default touch MESH_MIN/MESH_MAX; explicit values are preserved.
        """
        method = params.get("METHOD", None)
        if method is not None and method.lower() != "automatic":
            msg = (
                f"PROBE_METHOD=touch with {self.command_name} requires METHOD=automatic "
                f"(or omitted), but got METHOD={method!r}."
            )
            raise ValueError(msg)

        boundaries = self.probe.touch.boundaries
        mesh_min = params.get("MESH_MIN", f"{boundaries.min_x},{boundaries.min_y}")
        mesh_max = params.get("MESH_MAX", f"{boundaries.max_x},{boundaries.max_y}")

        new_params = self.gcode.clone_params(
            params,
            {
                "METHOD": "automatic",
                "MESH_MIN": mesh_min,
                "MESH_MAX": mesh_max,
            },
        )

        with self.probe.as_touch():
            self.fallback.run(new_params)
