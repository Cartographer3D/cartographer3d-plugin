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

    This class queues items received from background threads and schedules
    processing on the main reactor thread using register_async_callback.
    This ensures thread-safe access to shared state during processing.

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
        self._lock = threading.Lock()

    def queue_item(self, item: T) -> None:
        """
        Queue an item for processing on the main reactor thread.

        This method is thread-safe and can be called from any thread.
        It queues the item and schedules processing if not already scheduled.

        Parameters:
        -----------
        item : T
            The item to process.
        """
        with self._lock:
            self._pending_items.append(item)
            # Schedule processing on main thread if not already scheduled
            if not self._processing_scheduled:
                waketime = self._reactor.monotonic() + BATCH_INTERVAL
                self._processing_scheduled = True
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
