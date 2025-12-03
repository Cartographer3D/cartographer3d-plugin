from __future__ import annotations

from typing import TYPE_CHECKING, Callable, final, overload

from typing_extensions import override

from cartographer.interfaces.multiprocessing import Scheduler

if TYPE_CHECKING:
    from reactor import Reactor


@final
class KlipperScheduler(Scheduler):
    """Klipper-specific scheduler using the reactor pattern."""

    def __init__(self, reactor: Reactor) -> None:
        self._reactor = reactor

    @override
    def sleep(self, seconds: float) -> None:
        eventtime = self._reactor.monotonic()
        _ = self._reactor.pause(eventtime + seconds)

    @overload
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: None = None,
        poll_interval: float = 0.1,
    ) -> None: ...

    @overload
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float,
        poll_interval: float = 0.1,
    ) -> bool: ...

    @override
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> bool | None:
        eventtime = self._reactor.monotonic()
        end_time = eventtime + timeout if timeout is not None else float("inf")

        while not condition():
            eventtime = self._reactor.pause(eventtime + poll_interval)
            if eventtime >= end_time:
                return False

        return None if timeout is None else True
