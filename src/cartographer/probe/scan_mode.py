from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import AxisTwistCompensation, Endstop, HomingState, Position, ProbeMode, Sample
from cartographer.probe.scan_model import ScanModelSelectorMixin, TemperatureCompensationModel

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import (
        Configuration,
        ScanModelConfiguration,
    )
    from cartographer.interfaces.printer import Mcu, Toolhead
    from cartographer.stream import Session

logger = logging.getLogger(__name__)


class Model(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def z_offset(self) -> float: ...
    def distance_to_frequency(self, distance: float) -> float: ...
    def frequency_to_distance(self, frequency: float) -> float: ...


TRIGGER_DISTANCE = 2.0


@dataclass(frozen=True)
class ScanModeConfiguration:
    x_offset: float
    y_offset: float
    travel_speed: float
    probe_speed: float

    samples: int
    models: dict[str, ScanModelConfiguration]

    @staticmethod
    def from_config(config: Configuration):
        return ScanModeConfiguration(
            x_offset=config.general.x_offset,
            y_offset=config.general.y_offset,
            travel_speed=config.general.travel_speed,
            probe_speed=config.scan.probe_speed,
            samples=config.scan.samples,
            models=config.scan.models,
        )


class ScanMode(ScanModelSelectorMixin, ProbeMode, Endstop):
    """Implementation for Scan mode."""

    @property
    @override
    def offset(self) -> Position:
        return Position(self._config.x_offset, self._config.y_offset, self.probe_height)

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
        config: ScanModeConfiguration,
        temperature_compensation: TemperatureCompensationModel | None,
        axis_twist_compensation: AxisTwistCompensation | None,
    ) -> None:
        super().__init__(config.models)
        self._last_homing_time: float = 0.0
        self._toolhead: Toolhead = toolhead
        self._config: ScanModeConfiguration = config
        self.probe_height: float = TRIGGER_DISTANCE
        self._mcu: Mcu = mcu
        self._temperature_compensation: TemperatureCompensationModel | None = temperature_compensation
        self._axis_twist_compensation: AxisTwistCompensation | None = axis_twist_compensation

        self.last_z_result: float | None = None

    @override
    def get_compensation_model(self) -> TemperatureCompensationModel | None:
        return self._temperature_compensation

    @override
    def get_status(self, eventtime: float) -> dict[str, object]:
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

        dist = self.measure_distance()
        if dist > self.probe_height + 0.5:
            # Safely move downwards
            _ = self._toolhead.z_probing_move(self, speed=self._config.probe_speed)
        elif self._toolhead.get_position().z < self.probe_height:
            self._toolhead.move(z=self.probe_height, speed=self._config.probe_speed)
            self._toolhead.wait_moves()

        delta = self.probe_height - self.measure_distance()

        toolhead_pos = self._toolhead.get_position()
        dist = toolhead_pos.z + delta

        if math.isinf(dist):
            msg = "Toolhead stopped outside model range"
            raise RuntimeError(msg)

        z_result = dist
        if self._axis_twist_compensation:
            z_result += self._axis_twist_compensation.get_z_compensation_value(x=toolhead_pos.x, y=toolhead_pos.y)

        logger.info("probe at %.3f,%.3f is z=%.6f", toolhead_pos.x, toolhead_pos.y, z_result)
        self.last_z_result = z_result
        return self.last_z_result

    def measure_distance(
        self, *, time: float | None = None, min_sample_count: int | None = None, skip_count: int = 5
    ) -> float:
        model = self.get_model()

        min_sample_count = min_sample_count or self._config.samples
        time = time or self._toolhead.get_last_move_time()

        with self._mcu.start_session(lambda sample: sample.time >= time) as session:
            session.wait_for(lambda samples: len(samples) >= min_sample_count + skip_count)
        samples = session.get_items()[skip_count:]

        dist = float(
            np.median(
                [model.frequency_to_distance(sample.frequency, temperature=sample.temperature) for sample in samples]
            )
        )
        return dist

    def calculate_sample_distance(self, sample: Sample) -> float:
        model = self.get_model()
        return model.frequency_to_distance(sample.frequency, temperature=sample.temperature)

    @override
    def query_is_triggered(self, print_time: float) -> bool:
        if not self.has_model():
            return True  # No model loaded, assume triggered
        distance = self.measure_distance(time=print_time)
        return distance <= self.get_endstop_position()

    @override
    def get_endstop_position(self) -> float:
        return self.probe_height

    @override
    def home_start(self, print_time: float) -> object:
        # TODO: We should get the temperature from the coil
        # This is good enough, as we will correct the distance
        # with the measurement after homing triggers.
        trigger_frequency = self.get_model().distance_to_frequency(self.probe_height, temperature=40)
        return self._mcu.start_homing_scan(print_time, trigger_frequency)

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        if not homing_state.is_homing_z():
            return
        distance = self.measure_distance()
        if not math.isfinite(distance):
            msg = "Toolhead stopped outside model range"
            raise RuntimeError(msg)

        homing_state.set_z_homed_position(distance)
        self._last_homing_time = self._toolhead.get_last_move_time()

    @override
    def home_wait(self, home_end_time: float) -> float:
        return self._mcu.stop_homing(home_end_time)

    def start_session(self) -> Session[Sample]:
        time = self._toolhead.get_last_move_time()
        return self._mcu.start_session(lambda sample: sample.time >= time)
