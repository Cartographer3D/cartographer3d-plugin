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
