# https://github.com/Klipper3d/klipper/blob/master/klippy/webhooks.py

from typing import Callable, TypeVar

from gcode import CommandError

class WebRequestError(CommandError): ...

T = TypeVar("T")

class WebRequest:
    def get_client_connection(self) -> ClientConnection:
        """Get the client connection for the request."""
        ...

    def send(self, data: object) -> None:
        """Send response to the client."""
        ...

    def get_dict(self, key: str, default: dict[str, object]) -> dict[str, object]: ...

class ClientConnection:
    def is_closed(self) -> bool:
        """Check if the connection is closed."""
        ...

    def send(self, data: object) -> None:
        """Send data to the client."""
        ...

class WebHooks:
    def register_endpoint(self, endpoint: str, callback: Callable[[WebRequest], None]) -> None:
        """Register a webhooks endpoint."""
        ...
