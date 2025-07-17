from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Toolhead

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)


@dataclass
class CalibrationOptions:
    start: float | None
    end: float | None
    line: float | None


@dataclass
class CompensationResult:
    axis: Literal["x", "y"]
    start: float
    end: float
    values: list[float]


class AxisTwistCompensationAdapter(Protocol):
    move_height: float
    speed: float

    def clear_compensations(self, axis: Literal["x", "y"]) -> None: ...
    def apply_compensation(self, result: CompensationResult) -> None: ...
    def get_calibration_options(self, axis: Literal["x", "y"]) -> CalibrationOptions: ...


@final
class AxisTwistCompensationMacro(Macro):
    description = "Scan and touch to calculate axis twist compensation values."

    def __init__(
        self, probe: Probe, toolhead: Toolhead, adapter: AxisTwistCompensationAdapter, config: Configuration
    ) -> None:
        self.probe = probe
        self.toolhead = toolhead
        self.adapter = adapter
        self.config = config

    @override
    def run(self, params: MacroParams) -> None:
        axis = params.get("AXIS", default="x").lower()
        if axis not in ("x", "y"):
            msg = f"Invalid axis '{axis}'"
            raise RuntimeError(msg)
        sample_count = params.get_int("SAMPLE_COUNT", default=5)

        start_pos, end_pos, line_pos = self._get_calibration_positions(params, axis)

        self.adapter.clear_compensations(axis)
        try:
            self._calibrate(axis, sample_count, start_pos, end_pos, line_pos)
        except RuntimeError:
            logger.info("""
                Error during axis twist compensation calibration,
                existing compensation has been cleared.
                Restart firmware to restore.
                """)
            raise

    def _get_calibration_positions(self, params: MacroParams, axis: Literal["x", "y"]) -> tuple[float, float, float]:
        start_pos = params.get_float("START", default=None)
        end_pos = params.get_float("END", default=None)
        line_pos = params.get_float("LINE", default=None)

        options = self.adapter.get_calibration_options(axis)
        boundaries = self.probe.touch.boundaries
        if axis == "x":
            if start_pos is None:
                start_pos = options.start or boundaries.min_x
            if end_pos is None:
                end_pos = options.end or boundaries.max_x
            if line_pos is None:
                line_pos = options.line or round((boundaries.max_y + boundaries.min_y) / 2, 2)
            if not boundaries.is_within(x=start_pos, y=line_pos):
                msg = f"Start position {start_pos} is outside of touch boundaries"
                raise RuntimeError(msg)
            if not boundaries.is_within(x=end_pos, y=line_pos):
                msg = f"End position {end_pos} is outside of touch boundaries"
                raise RuntimeError(msg)

        elif axis == "y":
            if start_pos is None:
                start_pos = options.start or boundaries.min_y
            if end_pos is None:
                end_pos = options.end or boundaries.max_y
            if line_pos is None:
                line_pos = options.line or round((boundaries.max_x + boundaries.min_x) / 2, 2)
            if not boundaries.is_within(x=line_pos, y=start_pos):
                msg = f"Start position {start_pos} is outside of touch boundaries"
                raise RuntimeError(msg)
            if not boundaries.is_within(x=line_pos, y=end_pos):
                msg = f"End position {end_pos} is outside of touch boundaries"
                raise RuntimeError(msg)

        return start_pos, end_pos, line_pos

    def _calibrate(
        self,
        axis: Literal["x", "y"],
        sample_count: int,
        start_pos: float,
        end_pos: float,
        line_pos: float,
    ) -> None:
        step = (end_pos - start_pos) / (sample_count - 1)
        results: list[float] = []
        start_time = time.time()
        for i in range(sample_count):
            position = start_pos + i * step
            self._move_probe_to(axis, position, line_pos)
            scan = self.probe.perform_scan()
            self._move_nozzle_to(axis, position, line_pos)
            touch = self.probe.perform_touch()
            result = scan - touch
            logger.debug("Offset at %:.2f: %.6f", position, result)
            results.append(result)
        logger.debug("Axis twist measurements completed in %.2f seconds", time.time() - start_time)

        avg = float(np.mean(results))
        results = [avg - x for x in results]

        self.adapter.apply_compensation(CompensationResult(axis=axis, start=start_pos, end=end_pos, values=results))
        logger.info("""
            Axis twist compensation state has been saved
            for the current session.  The SAVE_CONFIG command will
            update the printer config file and restart the printer.
            """)
        logger.info(
            "Touch %s axis twist compensation calibration complete: mean z_offset: %.6f, offsets: (%s)",
            axis.upper(),
            avg,
            ", ".join(f"{s:.6f}" for s in results),
        )

    def _move_nozzle_to(self, axis: Literal["x", "y"], position: float, line_pos: float) -> None:
        self.toolhead.move(z=self.adapter.move_height, speed=self.adapter.speed)
        if axis == "x":
            self.toolhead.move(
                x=position,
                y=line_pos,
                speed=self.adapter.speed,
            )
        else:
            self.toolhead.move(
                x=line_pos,
                y=position,
                speed=self.adapter.speed,
            )

    def _move_probe_to(self, axis: Literal["x", "y"], position: float, calibration_axis: float) -> None:
        if axis == "x":
            self.toolhead.move(
                x=position - self.config.general.x_offset,
                y=calibration_axis - self.config.general.y_offset,
                speed=self.adapter.speed,
            )
        else:
            self.toolhead.move(
                x=calibration_axis - self.config.general.x_offset,
                y=position - self.config.general.y_offset,
                speed=self.adapter.speed,
            )
