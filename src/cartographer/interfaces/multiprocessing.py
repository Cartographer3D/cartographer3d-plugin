from typing import Callable, Protocol, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


class TaskExecutor(Protocol):
    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...
