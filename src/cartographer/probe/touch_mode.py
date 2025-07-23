from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Endstop, HomingState, Mcu, Position, ProbeMode, Toolhead
from cartographer.lib.statistics import compute_mad
from cartographer.probe.touch_model import TouchModelSelectorMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.configuration import Configuration, TouchModelConfiguration

logger = logging.getLogger(__name__)


MAD_TOLERANCE = 0.0054  # Statistically equivalent to 0.008mm stddev
RETRACT_DISTANCE = 2.0
MAX_TOUCH_TEMPERATURE = 155


@dataclass(frozen=True)
class TouchModeConfiguration:
    samples: int
    max_samples: int

    x_offset: float
    y_offset: float
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]

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

        # For negative offsets: increase min bounds, leave max bounds unchanged
        min_x = mesh_min_x - min(x_offset, 0)  # Only subtract negative offsets
        min_y = mesh_min_y - min(y_offset, 0)  # Only subtract negative offsets
        # For positive offsets: reduce max bounds, leave min bounds unchanged
        max_x = mesh_max_x - max(x_offset, 0)  # Only subtract positive offsets
        max_y = mesh_max_y - max(y_offset, 0)  # Only subtract positive offsets

        return TouchBoundaries(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )


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

    def __init__(self, mcu: Mcu, toolhead: Toolhead, config: TouchModeConfiguration) -> None:
        super().__init__(config.models)
        self._last_homing_time: float = 0.0
        self._toolhead: Toolhead = toolhead
        self._mcu: Mcu = mcu
        self._config: TouchModeConfiguration = config

        self.boundaries: TouchBoundaries = TouchBoundaries.from_config(config)

        self.last_z_result: float | None = None

    @override
    def get_status(self, eventtime: float) -> object:
        return {
            "current_model": self.get_model().name if self.has_model() else "none",
            "models": ", ".join(self._config.models.keys()),
            "last_z_result": self.last_z_result,
        }

    @override
    def perform_probe(self) -> float:
        if not self._toolhead.is_homed("z"):
            msg = "Z axis must be homed before probing"
            raise RuntimeError(msg)

        if self._toolhead.get_position().z < 5:
            self._toolhead.move(z=5, speed=5)
        self._toolhead.wait_moves()

        self.last_z_result = self._run_probe()
        return self.last_z_result

    def _run_probe(self) -> float:
        collected: list[float] = []
        touch_samples = self._config.samples
        touch_max_samples = self._config.max_samples
        logger.debug("Starting touch sequence for %d samples within %d touches...", touch_samples, touch_max_samples)

        for i in range(touch_max_samples):
            trigger_pos = self._perform_single_probe()
            collected.append(trigger_pos)
            logger.debug("Touch %d: %.6f", i + 1, trigger_pos)

            if len(collected) < touch_samples:
                continue

            valid_combo = self._find_valid_combination(collected, touch_samples)
            if valid_combo is None:
                continue

            self._log_sample_stats("Acceptable touch combination found", valid_combo)

            return float(np.median(valid_combo) if len(valid_combo) > 3 else np.mean(valid_combo))

        self._log_sample_stats("No valid touch combination found in samples", collected)
        msg = f"Unable to find {touch_samples} samples within tolerance after {touch_max_samples} touches"
        raise TouchError(msg)

    def _find_valid_combination(self, samples: list[float], size: int) -> tuple[float, ...] | None:
        for combo in combinations(samples, size):
            current_mad = compute_mad(combo)
            if current_mad <= MAD_TOLERANCE:
                return tuple(sorted(combo))
        return None

    def _perform_single_probe(self) -> float:
        model = self.get_model()
        if self._toolhead.get_position().z < RETRACT_DISTANCE:
            self._toolhead.move(z=RETRACT_DISTANCE, speed=5)
        self._toolhead.wait_moves()
        trigger_pos = self._toolhead.z_homing_move(self, speed=model.speed)
        pos = self._toolhead.get_position()
        self._toolhead.move(z=max(pos.z + RETRACT_DISTANCE, RETRACT_DISTANCE), speed=5)
        return trigger_pos - model.z_offset

    @override
    def home_start(self, print_time: float) -> object:
        model = self.get_model()
        if model.threshold <= 0:
            msg = "Threshold must positive"
            raise RuntimeError(msg)

        pos = self._toolhead.get_position()
        if not self.is_within_boundaries(x=pos.x, y=pos.y):
            msg = f"Position ({pos.x:.2f},{pos.y:.2f}) is outside of the touch boundaries"
            raise RuntimeError(msg)

        nozzle = self._toolhead.get_extruder_temperature()
        if nozzle.current > MAX_TOUCH_TEMPERATURE or nozzle.target > MAX_TOUCH_TEMPERATURE:
            msg = f"Nozzle temperature must be below {MAX_TOUCH_TEMPERATURE - 5:d}C"
            raise RuntimeError(msg)
        return self._mcu.start_homing_touch(print_time, model.threshold)

    def is_within_boundaries(self, *, x: float, y: float) -> bool:
        return self.boundaries.is_within(x=x, y=y)

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        if not homing_state.is_homing_z():
            return

        homing_state.set_z_homed_position(self.get_model().z_offset)
        self._last_homing_time = self._toolhead.get_last_move_time()

    @override
    def home_wait(self, home_end_time: float) -> float:
        return self._mcu.stop_homing(home_end_time)

    @override
    def query_is_triggered(self, print_time: float) -> bool:
        # Touch endstop is never in a triggered state.
        return False

    @override
    def get_endstop_position(self) -> float:
        return self.offset.z

    def _log_sample_stats(self, message: str, samples: Sequence[float]) -> None:
        max_v, min_v = max(samples), min(samples)
        mean = np.mean(samples)
        median = np.median(samples)
        range_v = max_v - min_v
        std_dev = np.std(samples)
        mad = compute_mad(samples)
        logger.debug(
            """
                %s: (%s)
                maximum %.6f, minimum %.6f, range %.6f,
                average %.6f, median %.6f, standard deviation %.6f,
                median absolute deviation %.6f
                """,
            message,
            ", ".join(f"{s:.6f}" for s in samples),
            max_v,
            min_v,
            range_v,
            mean,
            median,
            std_dev,
            mad,
        )
