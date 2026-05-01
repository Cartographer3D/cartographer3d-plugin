from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcu import (
        MCU,
        CommandQueryWrapper,
        CommandWrapper,
        TriggerDispatch,
    )
    from reactor import Reactor
    from stepper import MCU_stepper

    from cartographer.interfaces.printer import Position


class McuPlatform(Protocol):
    """Host firmware platform interface for CartographerMcu.

    Abstracts host-specific (Klipper/Kalico/SimpleAF) concerns away from the
    shared Cartographer firmware and data pipeline code.

    Each Klipper fork implements this protocol, encapsulating its own API
    surface and version-specific behaviours.  ``CartographerMcu`` consumes
    this protocol and never touches host internals directly.
    """

    # ------------------------------------------------------------------
    # Core host objects
    # ------------------------------------------------------------------

    @property
    def host_mcu(self) -> MCU:
        """The underlying host MCU object."""
        ...

    # ------------------------------------------------------------------
    # Trigger dispatch (homing synchronisation)
    # ------------------------------------------------------------------

    def create_trigger_dispatch(self) -> TriggerDispatch:
        """Create a trigger dispatch for homing synchronisation."""
        ...

    def add_stepper_to_dispatch(self, dispatch: TriggerDispatch, stepper: MCU_stepper) -> None:
        """Add a Z-axis stepper to the trigger dispatch."""
        ...

    def get_z_steppers(self) -> list[MCU_stepper]:
        """Return all active Z-axis steppers from kinematics."""
        ...

    # ------------------------------------------------------------------
    # Data response registration
    # ------------------------------------------------------------------

    def register_data_response(
        self,
        handler: Callable[..., None],
        fmt: str,
        name: str,
    ) -> None:
        """Register the firmware data response handler.

        Implementations select the correct registration method for their host
        firmware version (e.g. ``register_serial_response`` vs ``register_response``).
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def register_lifecycle_handlers(
        self,
        *,
        on_identify: Callable[[], None],
        on_connect: Callable[[], None],
        on_shutdown: Callable[[], None],
        on_reconnect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        """Register event handlers for MCU lifecycle events.

        ``on_reconnect`` and ``on_disconnect`` are only used by platforms that
        support non-critical MCU reconnection (e.g. Kalico).  Implementations
        that do not support reconnection ignore them.
        """
        ...

    def register_config_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for MCU config/initialisation."""
        ...

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    def is_disconnected(self) -> bool:
        """Return True if the MCU is in a non-critical disconnected state.

        Platforms without non-critical MCU support always return ``False``.
        """
        ...

    # ------------------------------------------------------------------
    # Version info
    # ------------------------------------------------------------------

    def get_mcu_version(self) -> str | None:
        """Return the firmware version string, or ``None`` if unavailable."""
        ...

    # ------------------------------------------------------------------
    # Position / kinematics
    # ------------------------------------------------------------------

    def get_requested_position(self, print_time: float) -> Position:
        """Get the commanded toolhead position at *print_time*.

        Uses stepper kinematics to calculate the position.  Must be called
        from the main reactor thread.
        """
        ...

    # ------------------------------------------------------------------
    # Clock conversion
    # ------------------------------------------------------------------

    def clock32_to_clock64(self, clock32: int) -> int:
        """Convert a 32-bit MCU clock to 64-bit."""
        ...

    def clock_to_print_time(self, clock64: int) -> float:
        """Convert a 64-bit MCU clock to print time."""
        ...

    # ------------------------------------------------------------------
    # Reactor
    # ------------------------------------------------------------------

    def get_reactor(self) -> Reactor:
        """Return the reactor instance for async scheduling."""
        ...

    def get_reactor_time(self) -> float:
        """Return the current reactor monotonic time."""
        ...

    # ------------------------------------------------------------------
    # Printer objects
    # ------------------------------------------------------------------

    def invoke_shutdown(self, msg: str) -> None:
        """Trigger a firmware shutdown with the given message."""
        ...

    # ------------------------------------------------------------------
    # MCU command delegation
    # ------------------------------------------------------------------

    def alloc_command_queue(self) -> object:
        """Allocate a command queue on the MCU."""
        ...

    def lookup_command(self, fmt: str, *, cq: object) -> CommandWrapper:
        """Look up a firmware command by format string."""
        ...

    def lookup_query_command(self, cmd: str, response: str, *, cq: object) -> CommandQueryWrapper:
        """Look up a query command with expected response format."""
        ...

    def get_constants(self) -> dict[str, float | int]:
        """Return firmware constants dict."""
        ...

    def mcu_error(self, msg: str) -> Exception:
        """Create an MCU-specific error."""
        ...
