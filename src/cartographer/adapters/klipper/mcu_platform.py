from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.adapters.klipper_like.mcu_platform import KlipperLikeMcuPlatform

if TYPE_CHECKING:
    from collections.abc import Callable

    from configfile import ConfigWrapper


@final
class KlipperMcuPlatform(KlipperLikeMcuPlatform):
    """McuPlatform implementation for stock Klipper (all versions).

    Handles API version differences (``register_serial_response`` vs
    ``register_response``) via a strategy stored at construction time.
    Reconnect/disconnect lifecycle events are not supported and are ignored.
    """

    def __init__(self, config: ConfigWrapper, mcu_name: str) -> None:
        super().__init__(config, mcu_name)

        # Strategy injection: normalise register_data_response to (handler, fmt, name)
        # at construction time so call sites never branch on the API version.
        if hasattr(self._host_mcu, "register_serial_response"):

            def _use_serial(handler: Callable[..., None], fmt: str, _name: str) -> None:
                _ = self._host_mcu.register_serial_response(handler, fmt)

            self._data_response_strategy: Callable[[Callable[..., None], str, str], None] = _use_serial
        else:

            def _use_response(handler: Callable[..., None], _fmt: str, name: str) -> None:
                self._host_mcu.register_response(handler, name)

            self._data_response_strategy = _use_response

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
        self._data_response_strategy(handler, fmt, name)

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
        del on_reconnect, on_disconnect  # Stock Klipper has no non-critical MCU events
        self._printer.register_event_handler("klippy:mcu_identify", on_identify)
        self._printer.register_event_handler("klippy:connect", on_connect)
        self._printer.register_event_handler("klippy:shutdown", on_shutdown)

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @override
    def is_disconnected(self) -> bool:
        return False
