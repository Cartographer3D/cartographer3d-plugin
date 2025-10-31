from __future__ import annotations

from cartographer.interfaces.printer import HomingAxis, Position


class MockToolhead:
    """Mock toolhead that tracks movements."""

    def __init__(self):
        self.position: Position = Position(0, 0, 0)
        self.moves: list[tuple[float, float]] = []  # (x, y)
        self.last_move_time: float = 0.0

    def move(self, x: float | None = None, y: float | None = None, z: float | None = None, speed: float | None = None):
        """Track movement commands."""
        del z, speed  # Not used in this mock
        new_x = x if x is not None else self.position.x
        new_y = y if y is not None else self.position.y
        self.last_move_time += 1  # Simulate time passing

        self.position = Position(new_x, new_y, 0)
        self.moves.append((new_x, new_y))

    def wait_moves(self):
        """Mock wait for moves."""
        pass

    def dwell(self, time: float):
        """Mock dwell."""
        self.last_move_time += time

    def get_last_move_time(self) -> float:
        return self.last_move_time

    def get_axis_limits(self, axis: HomingAxis) -> tuple[float, float]:
        """Mock axis limits - return arbitrary limits."""
        del axis
        return (0.0, 100.0)
