from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Callable, TypeVar, final

from typing_extensions import ParamSpec, override

from cartographer.interfaces.multiprocessing import TaskExecutor

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from reactor import Reactor

P = ParamSpec("P")
R = TypeVar("R")

WAIT_TIME = 0.1


@final
class KlipperMultiprocessingExecutor(TaskExecutor):
    """
    Execute tasks in a separate process using Klipper's reactor pattern.

    This executor runs callables in a child process and communicates results
    back through a pipe, integrating with Klipper's reactor for asynchronous
    waiting.
    """

    def __init__(self, reactor: Reactor) -> None:
        """
        Initialize the multiprocessing executor.

        Parameters
        ----------
        reactor : Reactor
            Klipper's event reactor for managing asynchronous operations.
        """
        self._reactor = reactor

    @override
    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Execute a function in a separate process and return its result.

        Parameters
        ----------
        fn : Callable[P, R]
            The function to execute in the child process.
        *args : P.args
            Positional arguments to pass to the function.
        **kwargs : P.kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        R
            The result of executing the function.

        Raises
        ------
        RuntimeError
            If the worker process terminates unexpectedly.
        Exception
            Any exception raised by the function in the child process.
        """

        def worker(child_conn: Connection) -> None:
            """
            Worker function executed in the child process.

            Executes the target function and sends the result or exception
            back through the pipe.
            """
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
        eventtime = self._reactor.monotonic()
        while proc.is_alive() and not parent_conn.poll():
            eventtime = self._reactor.pause(eventtime + WAIT_TIME)

        # Check if data is actually available
        if not parent_conn.poll():
            # Process died without sending data
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

        # Handle result
        if is_error:
            raise payload from None

        return payload
