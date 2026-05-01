"""
Async processor for handling MCU data on the main reactor thread.

This module provides thread-safe processing of MCU samples by queuing them
from the MCU response thread and processing them on the main reactor thread
where it's safe to access stepper positions and other shared state.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Generic, Protocol, TypeVar, final

logger = logging.getLogger(__name__)

T = TypeVar("T")

BATCH_INTERVAL = 0.1


class Reactor(Protocol):
    def monotonic(self) -> float: ...
    def register_async_callback(self, callback: Callable[[float], None], waketime: float = ...) -> None: ...


@final
class AsyncProcessor(Generic[T]):
    """
    Process items asynchronously on the main reactor thread.

    Items received from background threads are queued and processed on
    the main reactor thread via register_async_callback.

    By default, items are batched for up to ``BATCH_INTERVAL`` seconds
    to reduce reactor wake-ups. When immediate mode is enabled, items
    are dispatched without delay so that position lookups happen as
    close to the sample time as possible.

    Parameters:
    -----------
    reactor : Reactor
        The Klipper reactor instance.
    process_fn : Callable[[T], None]
        Function to process each item on main thread.
    """

    def __init__(
        self,
        reactor: Reactor,
        process_fn: Callable[[T], None],
    ) -> None:
        self._reactor = reactor
        self._process_fn = process_fn
        self._pending_items: list[T] = []
        self._processing_scheduled = False
        self._immediate = False
        self._lock = threading.Lock()

    def set_immediate(self, enabled: bool) -> None:
        """
        Enable or disable immediate dispatch mode.

        When enabled, items are dispatched to the reactor without delay
        instead of waiting for the batch interval. Use during streaming
        sessions where position accuracy matters.
        """
        with self._lock:
            self._immediate = enabled
            # Schedule immediate flush for any pending items — even if
            # a batched callback is already pending, this ensures items
            # don't wait for the full batch interval.
            if enabled and self._pending_items:
                self._processing_scheduled = True
                self._reactor.register_async_callback(self._process_pending_items)

    def queue_item(self, item: T) -> None:
        """
        Queue an item for processing on the main reactor thread.

        This method is thread-safe and can be called from any thread.

        Parameters:
        -----------
        item : T
            The item to process.
        """
        with self._lock:
            self._pending_items.append(item)
            if not self._processing_scheduled:
                self._processing_scheduled = True
                if self._immediate:
                    self._reactor.register_async_callback(self._process_pending_items)
                else:
                    waketime = self._reactor.monotonic() + BATCH_INTERVAL
                    self._reactor.register_async_callback(self._process_pending_items, waketime=waketime)

    def _process_pending_items(self, eventtime: float) -> None:
        """
        Process all pending items on the main reactor thread.

        This method runs on the main thread where it's safe to access
        stepper positions and other shared state.

        Parameters:
        -----------
        eventtime : float
            The current reactor event time.
        """
        del eventtime  # unused

        # Atomically grab all pending items and clear the scheduled flag
        with self._lock:
            pending = self._pending_items
            self._pending_items = []
            self._processing_scheduled = False

        # Process all items outside the lock
        for item in pending:
            try:
                self._process_fn(item)
            except Exception as e:
                logger.exception("Error processing queued item: %s", e)
