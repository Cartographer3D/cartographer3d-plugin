from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Callable, TypeVar, final

from typing_extensions import ParamSpec, override

from cartographer.interfaces.multiprocessing import TaskExecutor

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from cartographer.interfaces.multiprocessing import Scheduler

P = ParamSpec("P")
R = TypeVar("R")


@final
class MultiprocessingExecutor(TaskExecutor):
    """
    Execute tasks in a separate process.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    @override
    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        def worker(child_conn: Connection) -> None:
            try:
                result = fn(*args, **kwargs)
                child_conn.send((False, result))
            except Exception as e:
                child_conn.send((True, e))
            finally:
                child_conn.close()

        parent_conn, child_conn = multiprocessing.Pipe()
        proc = multiprocessing.Process(target=worker, args=(child_conn,), daemon=True)
        proc.start()

        # Wait for data to be available
        self._scheduler.wait_until(lambda: not proc.is_alive() or parent_conn.poll())

        # Check if data is actually available
        if not parent_conn.poll():
            proc.join()
            exit_code = proc.exitcode
            parent_conn.close()
            msg = f"Worker process terminated unexpectedly with exit code {exit_code}"
            raise RuntimeError(msg)

        # Receive result
        try:
            is_error, payload = parent_conn.recv()
        finally:
            parent_conn.close()
            proc.join()

        if is_error:
            raise payload from None

        return payload
