from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import runtime_checkable

from typing_extensions import Protocol, final


def _read_token() -> str:
    try:
        from cartographer.telemetry._token import TOKEN
    except ImportError:
        return ""
    return TOKEN  # type: ignore[no-any-return]


logger = logging.getLogger(__name__)


def get_telemetry_backend() -> TelemetryBackend | None:
    """Return a backend if a telemetry token is available, else ``None``."""
    token = _read_token()
    if token:
        return BetterStackBackend(token)
    return None


@runtime_checkable
class TelemetryBackend(Protocol):
    """Protocol for telemetry backends.

    Implementations must send the payload without blocking the caller
    and must never raise exceptions that could crash Klipper.
    """

    def send(self, payload: dict[str, object]) -> None: ...


@final
class BetterStackBackend:
    """Sends telemetry events to BetterStack Logs via HTTP POST."""

    _DEFAULT_ENDPOINT: str = "https://in.logs.betterstack.com"
    _TIMEOUT: int = 10

    def __init__(
        self,
        source_token: str,
        endpoint: str = _DEFAULT_ENDPOINT,
    ) -> None:
        self._source_token = source_token
        self._endpoint = endpoint

    def send(self, payload: dict[str, object]) -> None:
        """Fire-and-forget: send payload in a daemon thread. Never raises."""
        thread = threading.Thread(target=self._post, args=(payload,), daemon=True)
        thread.start()

    def _post(self, payload: dict[str, object]) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers={
                    "Authorization": f"Bearer {self._source_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                logger.debug("Telemetry: event sent (status=%s).", resp.status)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Telemetry: failed to send event: %s", exc)
