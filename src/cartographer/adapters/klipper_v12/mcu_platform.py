from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper_like.mcu_platform import KlipperLikeMcuPlatform
from cartographer.adapters.klipper_v12.trigger_dispatch import TriggerDispatch as V12TriggerDispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from cartographer.interfaces.mcu_platform import TriggerDispatch


@final
class KlipperV12McuPlatform(KlipperLikeMcuPlatform):
    """McuPlatform implementation for Klipper v0.12.

    Klipper v0.12 predates ``register_serial_response``; always uses
    ``register_response`` (name-based, not format-string-based).  It also
    predates the built-in ``TriggerDispatch``, so we supply the bundled
    backport from ``cartographer.adapters.klipper_v12.trigger_dispatch``.
    Reconnect/disconnect lifecycle events are not supported and are ignored.
    """

    # ------------------------------------------------------------------
    # Trigger dispatch
    # ------------------------------------------------------------------

    @override
    def create_trigger_dispatch(self) -> TriggerDispatch:
        return V12TriggerDispatch(self._host_mcu)

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
        del fmt  # V12 uses register_response with name, not format string
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
        del on_reconnect, on_disconnect  # Klipper v0.12 has no non-critical MCU events
        self._printer.register_event_handler("klippy:mcu_identify", on_identify)
        self._printer.register_event_handler("klippy:connect", on_connect)
        self._printer.register_event_handler("klippy:shutdown", on_shutdown)

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @override
    def is_disconnected(self) -> bool:
        return False
