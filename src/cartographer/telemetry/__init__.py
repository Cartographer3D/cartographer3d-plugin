from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING, final

from cartographer.telemetry.collector import collect_startup_data
from cartographer.telemetry.identity import get_or_create_anonymous_id

if TYPE_CHECKING:
    from cartographer.runtime.adapters import Adapters
    from cartographer.telemetry.backend import TelemetryBackend

logger = logging.getLogger(__name__)


@final
class TelemetryReporter:
    """Collects and forwards anonymous usage telemetry.

    All exceptions are caught internally so telemetry can never crash Klipper.
    """

    def __init__(self, backend: TelemetryBackend, adapters: Adapters) -> None:
        self._backend = backend
        self._adapters = adapters

    @cached_property
    def anonymous_id(self) -> str:
        """Persistent anonymous installation UUID."""
        return get_or_create_anonymous_id()

    def send_startup_event(self) -> None:
        """Collect and send startup telemetry."""
        try:
            payload = collect_startup_data(self._adapters, self.anonymous_id)
            self._backend.send(payload)
        except (OSError, TypeError, KeyError, AttributeError, ValueError) as exc:
            logger.warning("Telemetry: unexpected error in send_startup_event: %s", exc)

    def send_event(self, event: str, **fields: object) -> None:
        """Send a named event with arbitrary fields."""
        try:
            from cartographer import __version__

            payload: dict[str, object] = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "event": event,
                "anonymous_id": self.anonymous_id,
                "plugin_version": __version__,
                "mcu_version": self._adapters.mcu.get_mcu_version(),
                **fields,
            }
            self._backend.send(payload)
        except (OSError, TypeError, KeyError, AttributeError, ValueError) as exc:
            logger.warning("Telemetry: unexpected error in send_event(%s): %s", event, exc)
