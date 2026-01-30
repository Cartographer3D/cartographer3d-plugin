from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import (
    Endstop,
    HomingState,
    Mcu,
    Position,
    ProbeMode,
    Toolhead,
)
from cartographer.probe.touch_model import TouchModelSelectorMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.configuration import (
        Configuration,
        TouchModelConfiguration,
    )

logger = logging.getLogger(__name__)


TOUCH_ACCEL = 100
MAX_SAMPLE_RANGE = 0.010  # All samples must be within 10 microns
MAX_TOUCH_TEMPERATURE_EPSILON = 2


@dataclass(frozen=True)
class TouchModeConfiguration:
    samples: int
    max_samples: int

    x_offset: float
    y_offset: float
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    max_touch_temperature: int
    lift_speed: float

    retract_distance: float
    models: dict[str, TouchModelConfiguration]

    @staticmethod
    def from_config(config: Configuration):
        return TouchModeConfiguration(
            samples=config.touch.samples,
            max_samples=config.touch.max_samples,
            models=config.touch.models,
            x_offset=config.general.x_offset,
            y_offset=config.general.y_offset,
            mesh_min=config.bed_mesh.mesh_min,
            mesh_max=config.bed_mesh.mesh_max,
            max_touch_temperature=config.touch.max_touch_temperature,
            lift_speed=config.general.lift_speed,
            retract_distance=config.touch.retract_distance,
        )


class TouchError(RuntimeError):
    pass


@dataclass(frozen=True)
class TouchBoundaries:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def is_within(self, *, x: float, y: float) -> bool:
        epsilon = 0.01
        in_x_bounds = (self.min_x - epsilon) <= x <= (self.max_x + epsilon)
        in_y_bounds = (self.min_y - epsilon) <= y <= (self.max_y + epsilon)
        return in_x_bounds and in_y_bounds

    @staticmethod
    def from_config(config: TouchModeConfiguration) -> TouchBoundaries:
        mesh_min_x, mesh_min_y = config.mesh_min
        mesh_max_x, mesh_max_y = config.mesh_max
        x_offset = config.x_offset
        y_offset = config.y_offset

        min_x = mesh_min_x - min(x_offset, 0)
        min_y = mesh_min_y - min(y_offset, 0)
        max_x = mesh_max_x - max(x_offset, 0)
        max_y = mesh_max_y - max(y_offset, 0)

        return TouchBoundaries(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )


def compute_range(samples: Sequence[float]) -> float:
    """Compute the range (max - min) of samples."""
    if len(samples) < 2:
        return float("inf")
    return max(samples) - min(samples)


def find_best_subset(
    samples: Sequence[float],
    size: int,
) -> Sequence[float] | None:
    """Find the subset of samples with the smallest range."""
    result = heapq.nsmallest(
        1,
        combinations(samples, size),
        key=compute_range,
    )
    return result[0] if result else None


class TouchMode(TouchModelSelectorMixin, ProbeMode, Endstop):
    """Implementation for Survey Touch."""

    @property
    @override
    def offset(self) -> Position:
        return Position(0.0, 0.0, 0.0)

    @property
    @override
    def is_ready(self) -> bool:
        return self.has_model()

    @property
    @override
    def last_homing_time(self) -> float:
        return self._last_homing_time

    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: TouchModeConfiguration,
    ) -> None:
        super().__init__(config.models)
        self._last_homing_time: float = 0.0
        self._toolhead: Toolhead = toolhead
        self._mcu: Mcu = mcu
        self._config: TouchModeConfiguration = config

        self.boundaries: TouchBoundaries = TouchBoundaries.from_config(config)
        self.last_z_result: float | None = None

    @override
    def get_status(self, eventtime: float) -> dict[str, object]:
        return {
            "current_model": (self.get_model().name if self.has_model() else "none"),
            "models": ", ".join(self._config.models.keys()),
            "last_z_result": self.last_z_result,
        }

    @override
    def perform_probe(self) -> float:
        if not self._toolhead.is_homed("z"):
            msg = "Z axis must be homed before probing"
            raise RuntimeError(msg)

        if self._toolhead.get_position().z < self._config.retract_distance:
            self._toolhead.move(z=self._config.retract_distance, speed=self._config.lift_speed)
        self._toolhead.wait_moves()

        self.last_z_result = self._run_probe()
        return self.last_z_result

    def _run_probe(self) -> float:
        """
        Collect touch samples and find a consistent subset.

        Collects samples one at a time, checking after each if there's
        a subset of the required size where all samples are within
        the acceptable range.
        """
        collected: list[float] = []
        required_samples = self._config.samples
        max_samples = self._config.max_samples

        logger.debug(
            "Starting touch sequence for %d samples within %d touches...",
            required_samples,
            max_samples,
        )

        for i in range(max_samples):
            trigger_pos = self._perform_single_probe()
            collected.append(trigger_pos)
            logger.debug("Touch %d: %.4f", i + 1, trigger_pos)

            if len(collected) < required_samples:
                continue

            best = find_best_subset(collected, required_samples)
            if best is None:
                continue

            sample_range = compute_range(best)
            if sample_range > MAX_SAMPLE_RANGE:
                continue

            self._log_sample_stats("Acceptable samples found", best)
            return float(np.median(best))

        # Failed - log what we had
        self._log_sample_stats("No acceptable samples found", collected)
        best = find_best_subset(collected, required_samples)
        if best:
            self._log_sample_stats("Best subset was", best)

        msg = (
            f"Unable to find {required_samples:d} samples within {MAX_SAMPLE_RANGE:.3f}mm after {max_samples:d} touches"
        )
        raise TouchError(msg)

    def _perform_single_probe(self) -> float:
        model = self.get_model()
        if self._toolhead.get_position().z < self._config.retract_distance:
            self._toolhead.move(z=self._config.retract_distance, speed=self._config.lift_speed)
        self._toolhead.wait_moves()

        max_accel = self._toolhead.get_max_accel()
        self._toolhead.set_max_accel(TOUCH_ACCEL)
        try:
            trigger_pos = self._toolhead.z_probing_move(self, speed=model.speed)
        finally:
            self._toolhead.set_max_accel(max_accel)

        pos = self._toolhead.get_position()
        self._toolhead.move(
            z=max(pos.z + self._config.retract_distance, self._config.retract_distance),
            speed=5,
        )
        return trigger_pos - model.z_offset

    @override
    def home_start(self, print_time: float) -> object:
        model = self.get_model()
        if model.threshold <= 0:
            msg = "Threshold must positive"
            raise RuntimeError(msg)

        pos = self._toolhead.get_position()
        if not self.boundaries.is_within(x=pos.x, y=pos.y):
            msg = (
                f"Position ({pos.x:.2f}, {pos.y:.2f}) is outside touch boundaries. "
                f"Valid range: X=[{self.boundaries.min_x:.2f}, {self.boundaries.max_x:.2f}], "
                f"Y=[{self.boundaries.min_y:.2f}, {self.boundaries.max_y:.2f}]"
            )
            raise RuntimeError(msg)

        nozzle_temperature = max(self._toolhead.get_extruder_temperature())
        max_temp = self._config.max_touch_temperature
        if nozzle_temperature > max_temp + MAX_TOUCH_TEMPERATURE_EPSILON:
            msg = f"Nozzle temperature must be below {max_temp:d}C"
            raise RuntimeError(msg)
        return self._mcu.start_homing_touch(print_time, model.threshold)

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        if not homing_state.is_homing_z():
            return
        self._last_homing_time = self._toolhead.get_last_move_time()

    @override
    def home_wait(self, home_end_time: float) -> float:
        return self._mcu.stop_homing(home_end_time)

    @override
    def query_is_triggered(self, print_time: float) -> bool:
        return False

    @override
    def get_endstop_position(self) -> float:
        return self.offset.z

    def _log_sample_stats(
        self,
        message: str,
        samples: Sequence[float],
    ) -> None:
        if not samples:
            logger.debug("%s: (no samples)", message)
            return

        max_v = max(samples)
        min_v = min(samples)
        range_v = max_v - min_v
        mean = float(np.mean(samples))
        median = float(np.median(samples))

        logger.debug(
            "%s: (%s)\nrange %.4f (limit %.4f), min %.4f, max %.4f,\nmean %.4f, median %.4f",
            message,
            ", ".join(f"{s:.4f}" for s in samples),
            range_v,
            MAX_SAMPLE_RANGE,
            min_v,
            max_v,
            mean,
            median,
        )
