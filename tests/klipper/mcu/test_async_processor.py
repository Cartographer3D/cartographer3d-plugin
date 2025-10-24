from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cartographer.adapters.klipper.mcu.async_processor import AsyncProcessor

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeReactor:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[float], None]] = []
        self.current_time: float = 0.0

    def monotonic(self) -> float:
        return self.current_time

    def register_async_callback(self, callback: Callable[[float], None], waketime: float = 0) -> None:
        del waketime
        self.callbacks.append(callback)

    def run_pending_callbacks(self) -> None:
        callbacks_to_run = self.callbacks.copy()
        self.callbacks.clear()
        for callback in callbacks_to_run:
            callback(self.current_time)


class ProcessedItemsCollector:
    def __init__(self) -> None:
        self.items: list[int] = []

    def process(self, item: int) -> None:
        self.items.append(item)


class TestAsyncProcessor:
    @pytest.fixture
    def reactor(self) -> FakeReactor:
        return FakeReactor()

    @pytest.fixture
    def collector(self) -> ProcessedItemsCollector:
        return ProcessedItemsCollector()

    @pytest.fixture
    def processor(self, reactor: FakeReactor, collector: ProcessedItemsCollector) -> AsyncProcessor[int]:
        return AsyncProcessor[int](reactor, collector.process)

    def test_queue_single_item(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
        collector: ProcessedItemsCollector,
    ) -> None:
        """Test queueing and processing a single item."""
        processor.queue_item(42)

        # Should schedule a callback
        assert len(reactor.callbacks) == 1

        # Run the callback
        reactor.run_pending_callbacks()

        # Item should be processed
        assert collector.items == [42]

    def test_queue_multiple_items_batches(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
        collector: ProcessedItemsCollector,
    ) -> None:
        """Test that multiple items queued before callback runs are batched."""
        processor.queue_item(1)
        processor.queue_item(2)
        processor.queue_item(3)

        # Should only schedule one callback due to guard
        assert len(reactor.callbacks) == 1

        # Run the callback
        reactor.run_pending_callbacks()

        # All items should be processed
        assert collector.items == [1, 2, 3]

    def test_multiple_callback_cycles(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
        collector: ProcessedItemsCollector,
    ) -> None:
        """Test multiple cycles of queue -> process."""
        # First batch
        processor.queue_item(1)
        reactor.run_pending_callbacks()
        assert collector.items == [1]

        # Second batch
        processor.queue_item(2)
        processor.queue_item(3)
        reactor.run_pending_callbacks()
        assert collector.items == [1, 2, 3]

        # Third batch
        processor.queue_item(4)
        reactor.run_pending_callbacks()
        assert collector.items == [1, 2, 3, 4]

    def test_no_duplicate_scheduling(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
    ) -> None:
        """Test that scheduling guard prevents duplicate callbacks."""
        processor.queue_item(1)
        processor.queue_item(2)
        processor.queue_item(3)

        # Should only schedule once
        assert len(reactor.callbacks) == 1

    def test_thread_safe_queueing(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
        collector: ProcessedItemsCollector,
    ) -> None:
        """Test that queueing from multiple threads is safe."""
        threads: list[threading.Thread] = []
        items_per_thread = 100

        def queue_items(start: int) -> None:
            for i in range(items_per_thread):
                processor.queue_item(start + i)

        # Spawn multiple threads
        for t in range(10):
            thread = threading.Thread(target=queue_items, args=(t * items_per_thread,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Process all queued items
        reactor.run_pending_callbacks()

        # Should have all items (order may vary)
        expected_count = 10 * items_per_thread
        assert len(collector.items) == expected_count
        assert set(collector.items) == set(range(expected_count))

    def test_processing_exception_handling(self, reactor: FakeReactor) -> None:
        """Test that exceptions during processing don't crash the processor."""
        processed: list[int] = []

        def process_fn(item: int) -> None:
            if item == 2:
                msg = "Test exception"
                raise ValueError(msg)
            processed.append(item)

        processor = AsyncProcessor[int](reactor, process_fn)

        processor.queue_item(1)
        processor.queue_item(2)  # This will raise
        processor.queue_item(3)

        # Should not raise, exception should be caught
        reactor.run_pending_callbacks()

        # Items 1 and 3 should be processed, 2 should be skipped
        assert processed == [1, 3]

    def test_generic_type_support(self, reactor: FakeReactor) -> None:
        """Test that AsyncProcessor works with different types."""
        strings: list[str] = []

        def process_string(item: str) -> None:
            strings.append(item.upper())

        processor = AsyncProcessor[str](reactor, process_string)

        processor.queue_item("hello")
        processor.queue_item("world")

        reactor.run_pending_callbacks()

        assert strings == ["HELLO", "WORLD"]

    def test_complex_object_processing(self, reactor: FakeReactor) -> None:
        """Test processing complex objects."""
        from dataclasses import dataclass

        @dataclass
        class Sample:
            x: float
            y: float
            value: float

        results: list[float] = []

        def process_sample(sample: Sample) -> None:
            results.append(sample.x + sample.y + sample.value)

        processor = AsyncProcessor[Sample](reactor, process_sample)

        processor.queue_item(Sample(1.0, 2.0, 3.0))
        processor.queue_item(Sample(4.0, 5.0, 6.0))

        reactor.run_pending_callbacks()

        assert results == [6.0, 15.0]

    def test_empty_queue_after_processing(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
        collector: ProcessedItemsCollector,
    ) -> None:
        """Test that queue is empty after processing."""
        processor.queue_item(1)
        processor.queue_item(2)

        reactor.run_pending_callbacks()

        # Queue a new item - should schedule a new callback
        processor.queue_item(3)
        assert len(reactor.callbacks) == 1

        reactor.run_pending_callbacks()
        assert collector.items == [1, 2, 3]

    def test_callback_scheduling_flag_reset(
        self,
        processor: AsyncProcessor[int],
        reactor: FakeReactor,
    ) -> None:
        """Test that scheduling flag is properly reset after processing."""
        processor.queue_item(1)
        assert len(reactor.callbacks) == 1

        # Process
        reactor.run_pending_callbacks()
        assert len(reactor.callbacks) == 0

        # Queue new item - should schedule again
        processor.queue_item(2)
        assert len(reactor.callbacks) == 1

    def test_race_condition_queue_during_process(self, reactor: FakeReactor) -> None:
        """Test queueing items while processing is happening."""
        processed: list[int] = []
        processor: AsyncProcessor[int] | None = None

        def process_fn(item: int) -> None:
            processed.append(item)
            # Queue another item during processing
            if item == 1 and processor is not None:
                processor.queue_item(99)

        processor = AsyncProcessor[int](reactor, process_fn)

        processor.queue_item(1)
        processor.queue_item(2)

        reactor.run_pending_callbacks()

        # Original items should be processed
        assert 1 in processed
        assert 2 in processed

        # Item queued during processing should schedule new callback
        assert len(reactor.callbacks) == 1

        reactor.run_pending_callbacks()
        assert 99 in processed

    def test_integration_with_mock_reactor(self) -> None:
        """Test integration with a mock reactor that simulates real behavior."""
        mock_reactor = MagicMock()
        processed: list[str] = []

        def process_fn(item: str) -> None:
            processed.append(item)

        processor = AsyncProcessor[str](mock_reactor, process_fn)

        # Queue items
        processor.queue_item("test1")
        processor.queue_item("test2")

        # Verify register_async_callback was called
        assert mock_reactor.register_async_callback.call_count == 1

        # Get the callback that was registered
        callback = mock_reactor.register_async_callback.call_args[0][0]

        # Manually invoke the callback (simulating reactor)
        callback(123.45)

        # Items should be processed
        assert processed == ["test1", "test2"]

        # Queue another item after processing
        processor.queue_item("test3")
        assert mock_reactor.register_async_callback.call_count == 2


class TestAsyncProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_processing_clears_pending_list(self) -> None:
        """Test that processing atomically clears the pending list."""
        reactor = FakeReactor()
        collector = ProcessedItemsCollector()
        processor = AsyncProcessor[int](reactor, collector.process)

        processor.queue_item(1)
        processor.queue_item(2)

        # Get callback
        callback = reactor.callbacks[0]

        # Add more items before callback runs
        processor.queue_item(3)

        # Run callback
        callback(0.0)

        # All three items should be processed
        assert collector.items == [1, 2, 3]

        # Queue new item - should work normally
        processor.queue_item(4)
        reactor.run_pending_callbacks()

        assert collector.items == [1, 2, 3, 4]

    def test_no_scheduling_without_items(self) -> None:
        """Test that no callback is scheduled when queue is empty."""
        reactor = FakeReactor()
        collector = ProcessedItemsCollector()
        _ = AsyncProcessor[int](reactor, collector.process)

        # Don't queue anything
        assert len(reactor.callbacks) == 0
