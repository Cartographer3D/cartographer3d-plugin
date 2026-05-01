from __future__ import annotations

import threading
import time
from typing import Callable

import pytest
from typing_extensions import override

from cartographer.stream import Condition, Stream


class MockCondition(Condition):
    def __init__(self):
        self._condition: threading.Condition = threading.Condition()

    @override
    def notify_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    @override
    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self._condition:
            _ = self._condition.wait_for(predicate)


class MockStream(Stream[object]):
    @override
    def condition(self) -> Condition:
        return MockCondition()

    @override
    def start_streaming(self) -> None:
        pass

    @override
    def stop_streaming(self) -> None:
        pass


@pytest.fixture
def stream() -> Stream[object]:
    return MockStream()


class TestStream:
    def test_start_session(self, stream: Stream[int]) -> None:
        with stream.start_session() as session:
            stream.add_item(42)
        assert session.get_items() == [42]

    def test_start_session_with_condition(self, stream: Stream[int]) -> None:
        with stream.start_session(start_condition=lambda x: x == 2) as session:
            stream.add_item(1)
            stream.add_item(2)
            stream.add_item(3)
        assert session.get_items() == [2, 3]

    def test_wait_for(self, stream: Stream[int]) -> None:
        session = stream.start_session()

        def add_items():
            for i in range(5):
                stream.add_item(i)
                time.sleep(0.1)

        # Run adding items in a separate thread
        worker = threading.Thread(target=add_items)
        worker.start()

        # Wait until 5 items are collected
        session.wait_for(lambda numbers: len(numbers) >= 5)
        collected_numbers = session.get_items()

        assert collected_numbers == [0, 1, 2, 3, 4]  # Items should be collected

        stream.end_session(session)
        worker.join()  # Ensure thread has finished before exiting


class TestSessionAbort:
    def test_session_abort_wakes_waiter_with_runtime_error(self, stream: Stream[object]) -> None:
        """Aborting a session causes wait_for to raise RuntimeError."""
        session = stream.start_session()

        def abort_soon() -> None:
            time.sleep(0.05)
            session.abort()

        worker = threading.Thread(target=abort_soon)
        worker.start()

        with pytest.raises(RuntimeError, match="disconnected"):
            session.wait_for(lambda items: len(items) >= 100)  # never-true condition

        worker.join()

    def test_abort_already_set_raises_immediately(self, stream: Stream[object]) -> None:
        """If abort() was called before wait_for, wait_for raises without blocking."""
        session = stream.start_session()
        session.abort()

        with pytest.raises(RuntimeError, match="disconnected"):
            session.wait_for(lambda items: False)


class TestAbortAllSessions:
    def test_stream_abort_all_sessions_aborts_each(self, stream: Stream[object]) -> None:
        """abort_all_sessions aborts every active session."""
        session_a = stream.start_session()
        session_b = stream.start_session()

        stream.abort_all_sessions()

        with pytest.raises(RuntimeError):
            session_a.wait_for(lambda items: False)

        with pytest.raises(RuntimeError):
            session_b.wait_for(lambda items: False)

    def test_stream_abort_all_sessions_empty(self, stream: Stream[object]) -> None:
        """abort_all_sessions on a stream with no sessions does not raise."""
        stream.abort_all_sessions()  # Should not raise
