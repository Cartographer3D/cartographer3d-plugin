from __future__ import annotations

from typing import Callable, Protocol, TypeVar, overload

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


class TaskExecutor(Protocol):
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
        ...


class Scheduler(Protocol):
    """Abstract interface for time-based operations."""

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...

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

    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> bool | None:
        """
        Wait until a condition becomes true or timeout expires.

        Parameters
        ----------
        condition : Callable[[], bool]
            A callable that returns True when the wait should end.
        timeout : float | None
            Maximum time to wait in seconds. None means wait indefinitely.
        poll_interval : float
            Time between condition checks in seconds.

        Returns
        -------
        bool | None
            If timeout is None, returns None (waits indefinitely).
            If timeout is specified, returns True if condition was met,
            False if timeout expired.
        """
        ...
