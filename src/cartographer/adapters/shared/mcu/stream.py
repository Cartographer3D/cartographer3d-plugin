from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, final

import greenlet
from typing_extensions import override

from cartographer.stream import Condition, Session, Stream

if TYPE_CHECKING:
    from reactor import Reactor


@final
class KlipperCondition(Condition):
    """The Klipper equivalent of [threading.Condition](https://docs.python.org/3/library/threading.html#condition-objects)"""

    def __init__(self, reactor: Reactor):
        self.reactor = reactor
        self.waiting: list[greenlet.greenlet] = []

    @override
    def notify_all(self):
        for wait in self.waiting:
            self.reactor.update_timer(wait.timer, self.reactor.NOW)

    @override
    def wait_for(self, predicate: Callable[[], bool]) -> None:
        wait = greenlet.getcurrent()
        self.waiting.append(wait)
        while True:
            if predicate():
                break
            _ = self.reactor.pause(self.reactor.NEVER)
        self.waiting.remove(wait)


T = TypeVar("T")


class KlipperStreamMcu(Protocol):
    def start_streaming(self) -> None:
        """Used to ask the MCU to start sending data."""
        ...

    def stop_streaming(self) -> None:
        """Stop the MCU from sending data.
        Will be called when the last session ends.
        """
        ...


@final
class KlipperStream(Stream[T]):
    def __init__(
        self,
        mcu: KlipperStreamMcu,
        reactor: Reactor,
        smoothing_fn: Callable[[T], T] | None = None,
    ):
        self.reactor = reactor
        self.mcu = mcu
        super().__init__(smoothing_fn)

    @override
    def condition(self) -> Condition:
        return KlipperCondition(self.reactor)

    @override
    def start_session(self, start_condition: Callable[[T], bool] | None = None) -> Session[T]:
        if len(self.sessions) == 0:
            self.mcu.start_streaming()
        return super().start_session(start_condition)

    @override
    def end_session(self, session: Session[T]) -> None:
        super().end_session(session)
        if len(self.sessions) == 0:
            self.mcu.stop_streaming()
