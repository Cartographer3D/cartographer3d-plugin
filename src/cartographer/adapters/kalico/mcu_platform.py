from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper_like.mcu_platform import KlipperLikeMcuPlatform

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcu import MCU

    class KalicoMCU(MCU):
        """Typing extension for Kalico's MCU which adds non-critical disconnect support."""

        non_critical_disconnected: bool  # pyright: ignore[reportUninitializedInstanceVariable]

        def get_non_critical_reconnect_event_name(self) -> str: ...

        def get_non_critical_disconnect_event_name(self) -> str: ...


@final
class KalicoMcuPlatform(KlipperLikeMcuPlatform):
    """McuPlatform implementation for Kalico.
    Kalico diverged from Klipper before ``register_serial_response`` was added,
    so it uses ``register_response`` unconditionally.

    - The MCU exposes ``non_critical_disconnected`` for disconnection state.
    - Non-critical reconnect/disconnect events are available via
      ``get_non_critical_reconnect_event_name()`` /
      ``get_non_critical_disconnect_event_name()``.
    """

    @property
    def _kalico_mcu(self) -> KalicoMCU:
        return self._host_mcu  # pyright: ignore[reportReturnType]

    # ------------------------------------------------------------------
    # Data response registration
    # ------------------------------------------------------------------

    @override
    def register_data_response(
        self,
        handler: Callable[..., None],
        fmt: str,
        name: str,
    ) -> None:
        del fmt  # Kalico uses register_response with name, not format string
        self._host_mcu.register_response(handler, name)

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    @override
    def register_lifecycle_handlers(
        self,
        *,
        on_identify: Callable[[], None],
        on_connect: Callable[[], None],
        on_shutdown: Callable[[], None],
        on_reconnect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        self._printer.register_event_handler("klippy:mcu_identify", on_identify)
        self._printer.register_event_handler("klippy:connect", on_connect)
        self._printer.register_event_handler("klippy:shutdown", on_shutdown)

        if on_reconnect is not None:
            reconnect_event = self._kalico_mcu.get_non_critical_reconnect_event_name()
            self._printer.register_event_handler(reconnect_event, on_reconnect)

        if on_disconnect is not None:
            disconnect_event = self._kalico_mcu.get_non_critical_disconnect_event_name()
            self._printer.register_event_handler(disconnect_event, on_disconnect)

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @override
    def is_disconnected(self) -> bool:
        return self._kalico_mcu.non_critical_disconnected
