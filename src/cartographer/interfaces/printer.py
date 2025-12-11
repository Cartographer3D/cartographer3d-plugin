from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from cartographer.stream import Session

HomingAxis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_list(self) -> list[float]:
        return list(self.as_tuple())


class HomingState(Protocol):
    def is_homing_z(self) -> bool:
        """Check if the z axis is currently being homed."""
        ...

    def set_z_homed_position(self, position: float) -> None:
        """Set the homed position for the z axis."""
        ...


class Endstop(Protocol):
    """Endstop interface for homing operations."""

    def query_is_triggered(self, print_time: float) -> bool:
        """Return true if endstop is currently triggered"""
        ...

    def home_start(self, print_time: float) -> object:
        """Start the homing process"""
        ...

    def home_wait(self, home_end_time: float) -> float:
        """Wait for homing to complete"""
        ...

    def on_home_end(self, homing_state: HomingState) -> None:
        """To be called when the homing process is complete"""
        ...

    def get_endstop_position(self) -> float:
        """The position of the endstop on the rail"""
        ...


@dataclass(frozen=True)
class Sample:
    frequency: float
    time: float
    position: Position | None
    temperature: float
    raw_count: int


class CoilCalibrationReference(NamedTuple):
    min_frequency: float
    min_frequency_temperature: float


class Mcu(Protocol):
    def start_homing_scan(self, print_time: float, frequency: float) -> object: ...
    def start_homing_touch(self, print_time: float, threshold: int) -> object: ...
    def stop_homing(self, home_end_time: float) -> float: ...
    def start_session(self, start_condition: Callable[[Sample], bool] | None = None) -> Session[Sample]: ...
    def register_callback(self, callback: Callable[[Sample], None]) -> None: ...
    def unregister_callback(self, callback: Callable[[Sample], None]) -> None: ...
    def get_current_time(self) -> float: ...
    def get_coil_reference(self) -> CoilCalibrationReference: ...
    def get_status(self, eventtime: float) -> dict[str, object]: ...
    def get_mcu_version(self) -> str: ...
    def get_last_sample(self) -> Sample | None: ...


class MacroParams(Protocol):
    @overload
    def get(self, name: str, default: str = ...) -> str: ...
    @overload
    def get(self, name: str, default: None) -> str | None: ...
    @overload
    def get_float(
        self, name: str, default: float = ..., *, above: float = ..., minval: float = ..., maxval: float = ...
    ) -> float: ...
    @overload
    def get_float(
        self, name: str, default: None, *, above: float = ..., minval: float = ..., maxval: float = ...
    ) -> float | None: ...
    def get_int(
        self,
        name: str,
        default: int = ...,
        *,
        minval: int = ...,
        maxval: int = ...,
    ) -> int: ...


@runtime_checkable
class SupportsFallbackMacro(Protocol):
    def set_fallback_macro(self, macro: Macro) -> None: ...


class Macro(Protocol):
    description: str

    def run(self, params: MacroParams) -> None: ...


class ProbeMode(Protocol):
    @property
    def offset(self) -> Position: ...
    @property
    def is_ready(self) -> bool: ...
    @property
    def last_homing_time(self) -> float: ...
    def get_status(self, eventtime: float) -> object: ...
    def perform_probe(self) -> float: ...


class TemperatureStatus(NamedTuple):
    current: float
    target: float


class GCodeDispatch(Protocol):
    def run_gcode(self, script: str) -> None:
        """Run the given gcode script."""
        ...


class AxisTwistCompensation(Protocol):
    def get_z_compensation_value(self, *, x: float, y: float) -> float: ...


class Toolhead(Protocol):
    def get_last_move_time(self) -> float:
        """Returns the last time the toolhead moved."""
        ...

    def wait_moves(self) -> None:
        """Wait for all moves to complete."""
        ...

    def get_position(self) -> Position:
        """Get the currently commanded position of the toolhead."""
        ...

    def move(self, *, x: float | None = None, y: float | None = None, z: float | None = None, speed: float) -> None:
        """Move to requested position."""
        ...

    def is_homed(self, axis: HomingAxis) -> bool:
        """Check if axis is homed."""
        ...

    def get_gcode_z_offset(self) -> float:
        """Returns currently applied gcode offset for the z axis."""
        ...

    def z_probing_move(self, endstop: Endstop, *, speed: float) -> float:
        """Starts probing move towards the given endstop."""
        ...

    def z_home_end(self, endstop: Endstop) -> None:
        """Informs the modules that homing has ended on the z axis."""
        ...

    def set_z_position(self, z: float) -> None:
        """Set the z position of the toolhead."""
        ...

    def get_axis_limits(self, axis: HomingAxis) -> tuple[float, float]:
        """Get the limits of an axis."""
        ...

    def manual_probe(self, finalize_callback: Callable[[Position | None], None]) -> None:
        """Start a manual probe."""
        ...

    def clear_z_homing_state(self) -> None:
        """Clears z homing state"""
        ...

    def dwell(self, seconds: float) -> None:
        """Dwell for the given number of seconds."""
        ...

    def get_extruder_temperature(self) -> TemperatureStatus:
        """Get the current and target temperature of the extruder."""
        ...

    def get_max_accel(self) -> float:
        """Get the current maximum accel"""
        ...

    def set_max_accel(self, accel: float) -> None:
        """Set the current maximum accel"""
        ...
