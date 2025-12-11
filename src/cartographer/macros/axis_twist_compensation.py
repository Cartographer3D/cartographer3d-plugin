from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import AxisTwistCompensation, Macro, MacroParams, Toolhead

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe import Probe
    from cartographer.probe.touch_mode import TouchBoundaries

logger = logging.getLogger(__name__)


@dataclass
class CalibrationOptions:
    start: float | None
    end: float | None
    line: float | None


@dataclass(frozen=True)
class ResolvedCalibrationOptions:
    start: float
    end: float
    line: float


@dataclass
class CompensationResult:
    axis: Literal["x", "y"]
    start: float
    end: float
    values: list[float]


class AxisTwistCompensationAdapter(AxisTwistCompensation, Protocol):
    move_height: float
    speed: float

    def clear_compensations(self, axis: Literal["x", "y"]) -> None: ...
    def apply_compensation(self, result: CompensationResult) -> None: ...
    def get_calibration_options(self, axis: Literal["x", "y"]) -> CalibrationOptions: ...


def _options_from_boundaries(boundaries: TouchBoundaries, axis: Literal["x", "y"]) -> ResolvedCalibrationOptions:
    """Build fully resolved options from touch boundaries."""
    if axis == "x":
        return ResolvedCalibrationOptions(
            start=boundaries.min_x,
            end=boundaries.max_x,
            line=round((boundaries.max_y + boundaries.min_y) / 2, 2),
        )
    return ResolvedCalibrationOptions(
        start=boundaries.min_y,
        end=boundaries.max_y,
        line=round((boundaries.max_x + boundaries.min_x) / 2, 2),
    )


def _resolve_options(options: CalibrationOptions, fallback: ResolvedCalibrationOptions) -> ResolvedCalibrationOptions:
    """Merge optional calibration options with resolved fallbacks."""
    return ResolvedCalibrationOptions(
        start=options.start if options.start is not None else fallback.start,
        end=options.end if options.end is not None else fallback.end,
        line=options.line if options.line is not None else fallback.line,
    )


def _apply_overrides(params: MacroParams, options: ResolvedCalibrationOptions) -> ResolvedCalibrationOptions:
    """Apply macro parameter overrides to resolved options."""
    start = params.get_float("START", default=None)
    end = params.get_float("END", default=None)
    line = params.get_float("LINE", default=None)

    return ResolvedCalibrationOptions(
        start=start if start is not None else options.start,
        end=end if end is not None else options.end,
        line=line if line is not None else options.line,
    )


def _validate_options(
    options: ResolvedCalibrationOptions,
    boundaries: TouchBoundaries,
    axis: Literal["x", "y"],
) -> None:
    """Validate calibration options against touch boundaries."""
    if axis == "x":
        _validate_point(boundaries, x=options.start, y=options.line, label="Start")
        _validate_point(boundaries, x=options.end, y=options.line, label="End")
    else:
        _validate_point(boundaries, x=options.line, y=options.start, label="Start")
        _validate_point(boundaries, x=options.line, y=options.end, label="End")


def _validate_point(boundaries: TouchBoundaries, *, x: float, y: float, label: str) -> None:
    """Validate a single point is within touch boundaries."""
    if boundaries.is_within(x=x, y=y):
        return

    msg = (
        f"{label} position ({x:.2f}, {y:.2f}) is outside touch boundaries. "
        f"Valid range: X=[{boundaries.min_x:.2f}, {boundaries.max_x:.2f}], "
        f"Y=[{boundaries.min_y:.2f}, {boundaries.max_y:.2f}]"
    )
    raise RuntimeError(msg)


@final
class AxisTwistCompensationMacro(Macro):
    description = "Scan and touch to calculate axis twist compensation values."

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: AxisTwistCompensationAdapter,
        config: Configuration,
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
        use_touch_boundaries = params.get_int("USE_TOUCH_BOUNDARIES", default=0) != 0

        boundaries = self.probe.touch.boundaries
        boundary_options = _options_from_boundaries(boundaries, axis)

        if use_touch_boundaries:
            base_options = boundary_options
        else:
            adapter_options = self.adapter.get_calibration_options(axis)
            base_options = _resolve_options(adapter_options, boundary_options)

        options = _apply_overrides(params, base_options)
        _validate_options(options, boundaries, axis)

        self.adapter.clear_compensations(axis)
        try:
            self._calibrate(axis, sample_count, options)
        except RuntimeError:
            logger.info(
                "Error during axis twist compensation calibration, "
                "existing compensation has been cleared. "
                "Restart firmware to restore."
            )
            raise

    def _calibrate(self, axis: Literal["x", "y"], sample_count: int, options: ResolvedCalibrationOptions) -> None:
        step = (options.end - options.start) / (sample_count - 1)
        results: list[float] = []
        start_time = time.time()

        for i in range(sample_count):
            position = options.start + i * step
            self._move_probe_to(axis, position, options.line)
            scan = self.probe.perform_scan()
            self._move_nozzle_to(axis, position, options.line)
            touch = self.probe.perform_touch()
            result = scan - touch
            logger.debug("Offset at %.2f: %.6f", position, result)
            results.append(result)

        logger.debug("Axis twist measurements completed in %.2f seconds", time.time() - start_time)

        avg = float(np.mean(results))
        results = [avg - x for x in results]

        self.adapter.apply_compensation(
            CompensationResult(
                axis=axis,
                start=options.start,
                end=options.end,
                values=results,
            )
        )
        logger.info(
            "Axis twist compensation state has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer."
        )
        logger.info(
            "Touch %s axis twist compensation calibration complete: mean z_offset: %.6f\noffsets: (%s)",
            axis.upper(),
            avg,
            ", ".join(f"{s:.6f}" for s in results),
        )

    def _move_nozzle_to(self, axis: Literal["x", "y"], position: float, line_pos: float) -> None:
        self.toolhead.move(z=self.adapter.move_height, speed=self.adapter.speed)
        if axis == "x":
            self.toolhead.move(x=position, y=line_pos, speed=self.adapter.speed)
        else:
            self.toolhead.move(x=line_pos, y=position, speed=self.adapter.speed)

    def _move_probe_to(self, axis: Literal["x", "y"], position: float, line_pos: float) -> None:
        x_offset = self.config.general.x_offset
        y_offset = self.config.general.y_offset
        if axis == "x":
            self.toolhead.move(x=position - x_offset, y=line_pos - y_offset, speed=self.adapter.speed)
        else:
            self.toolhead.move(x=line_pos - x_offset, y=position - y_offset, speed=self.adapter.speed)
