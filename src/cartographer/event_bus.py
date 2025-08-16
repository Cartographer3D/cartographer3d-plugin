from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar


@dataclass
class Event(Protocol): ...


T = TypeVar("T", bound=Event)

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._subscriptions: dict[type, list[Callable[[Any], None]]] = {}  # pyright: ignore[reportExplicitAny]

    def subscribe(self, event_type: type[T], callback: Callable[[T], None]):
        """Subscribe to events of a specific type."""
        self._subscriptions.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: type[T], callback: Callable[[T], None]):
        """Unsubscribe from events of a specific type."""
        if event_type in self._subscriptions:
            self._subscriptions[event_type].remove(callback)

    def publish(self, event: object):
        """Publish an event to all subscribers of its type."""
        event_type = type(event)
        logger.debug("Publishing event %s", event)

        for callback in self._subscriptions.get(event_type, []) + self._subscriptions.get(Event, []):
            try:
                callback(event)
            except Exception as e:
                logger.error("Error in event callback: %s", e)
