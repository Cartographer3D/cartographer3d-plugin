from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import mcu as _mcu_module
from mcu import TriggerDispatch as _KlipperTriggerDispatch

from cartographer.interfaces.printer import Position

if TYPE_CHECKING:
    from collections.abc import Callable

    from configfile import ConfigWrapper
    from kinematics import Kinematics
    from klippy import Printer
    from mcu import (
        MCU,
        CommandQueryWrapper,
        CommandWrapper,
        TriggerDispatch,
    )
    from reactor import Reactor
    from stepper import MCU_stepper
    from toolhead import ToolHead


class KlipperLikeMcuPlatform(ABC):
    """Shared base for Klipper-like McuPlatform implementations.

    Encapsulates the common code that is identical between
    ``KlipperMcuPlatform`` and ``KalicoMcuPlatform``.  Subclasses
    implement the three divergent methods.
    """

    def __init__(self, config: ConfigWrapper, mcu_name: str) -> None:
        self._printer: Printer = config.get_printer()
        self._host_mcu: MCU = _mcu_module.get_printer_mcu(self._printer, mcu_name)

    # ------------------------------------------------------------------
    # Core host objects
    # ------------------------------------------------------------------

    @property
    def host_mcu(self) -> MCU:
        return self._host_mcu

    # ------------------------------------------------------------------
    # Trigger dispatch
    # ------------------------------------------------------------------

    def create_trigger_dispatch(self) -> TriggerDispatch:
        return _KlipperTriggerDispatch(self._host_mcu)

    def add_stepper_to_dispatch(self, dispatch: TriggerDispatch, stepper: MCU_stepper) -> None:
        dispatch.add_stepper(stepper)

    def get_z_steppers(self) -> list[MCU_stepper]:
        return [s for s in self._kinematics.get_steppers() if s.is_active_axis("z")]

    # ------------------------------------------------------------------
    # Kinematics (lazy)
    # ------------------------------------------------------------------

    @cached_property
    def _toolhead(self) -> ToolHead:
        return self._printer.lookup_object("toolhead")

    @cached_property
    def _kinematics(self) -> Kinematics:
        return self._toolhead.get_kinematics()

    # ------------------------------------------------------------------
    # Data response registration (divergent — subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def register_data_response(
        self,
        handler: Callable[..., None],
        fmt: str,
        name: str,
    ) -> None: ...

    # ------------------------------------------------------------------
    # Lifecycle events (divergent — subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def register_lifecycle_handlers(
        self,
        *,
        on_identify: Callable[[], None],
        on_connect: Callable[[], None],
        on_shutdown: Callable[[], None],
        on_reconnect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None: ...

    def register_config_callback(self, callback: Callable[[], None]) -> None:
        self._host_mcu.register_config_callback(callback)

    # ------------------------------------------------------------------
    # Connection state (divergent — subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def is_disconnected(self) -> bool: ...

    # ------------------------------------------------------------------
    # Version info
    # ------------------------------------------------------------------

    def get_mcu_version(self) -> str | None:
        return self._host_mcu.get_status().get("mcu_version")

    # ------------------------------------------------------------------
    # Position / kinematics
    # ------------------------------------------------------------------

    def get_requested_position(self, print_time: float) -> Position:
        kinematics = self._kinematics
        stepper_pos = {
            stepper.get_name(): stepper.mcu_to_commanded_position(stepper.get_past_mcu_position(print_time))
            for stepper in kinematics.get_steppers()
        }
        position = kinematics.calc_position(stepper_pos)
        return Position(x=position[0], y=position[1], z=position[2])

    # ------------------------------------------------------------------
    # Clock conversion
    # ------------------------------------------------------------------

    def clock32_to_clock64(self, clock32: int) -> int:
        return self._host_mcu.clock32_to_clock64(clock32)

    def clock_to_print_time(self, clock64: int) -> float:
        return self._host_mcu.clock_to_print_time(clock64)

    # ------------------------------------------------------------------
    # Reactor
    # ------------------------------------------------------------------

    def get_reactor(self) -> Reactor:
        return self._host_mcu.get_printer().get_reactor()

    def get_reactor_time(self) -> float:
        return self.get_reactor().monotonic()

    # ------------------------------------------------------------------
    # Printer objects
    # ------------------------------------------------------------------

    def invoke_shutdown(self, msg: str) -> None:
        self._printer.invoke_shutdown(msg)

    # ------------------------------------------------------------------
    # MCU command delegation
    # ------------------------------------------------------------------

    def alloc_command_queue(self) -> object:
        return self._host_mcu.alloc_command_queue()

    def lookup_command(self, fmt: str, *, cq: object) -> CommandWrapper:
        return self._host_mcu.lookup_command(fmt, cq=cq)

    def lookup_query_command(self, cmd: str, response: str, *, cq: object) -> CommandQueryWrapper:
        return self._host_mcu.lookup_query_command(cmd, response, cq=cq)  # cq is opaque; came from alloc_command_queue

    def get_constants(self) -> dict[str, float | int]:
        return self._host_mcu.get_constants()

    def mcu_error(self, msg: str) -> Exception:
        return self._host_mcu.error(msg)
