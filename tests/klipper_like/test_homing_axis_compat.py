from __future__ import annotations

from cartographer.adapters.klipper_like.axis_compat import uses_string_homing_axes


class StringAxesToolhead:
    def set_position(self, newpos: list[float], homing_axes: str = "") -> None:
        del newpos, homing_axes


class IntegerAxesToolhead:
    def set_position(self, newpos: list[float], homing_axes: tuple[int, ...] = ()) -> None:
        del newpos, homing_axes


def test_detects_string_homing_axes() -> None:
    assert uses_string_homing_axes(StringAxesToolhead())


def test_detects_integer_homing_axes() -> None:
    assert not uses_string_homing_axes(IntegerAxesToolhead())
