from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Endstop, HomingState, Position, ProbeMode, Sample

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
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

    samples: int

    @staticmethod
    def from_config(config: Configuration):
        return ScanModeConfiguration(
            x_offset=config.general.x_offset,
            y_offset=config.general.y_offset,
            travel_speed=config.general.travel_speed,
            samples=config.scan.samples,
        )


class ScanMode(ProbeMode, Endstop):
    """Implementation for Scan mode."""

    def get_model(self) -> Model:
        if self.model is None:
            msg = "no scan model loaded"
            raise RuntimeError(msg)
        return self.model

    @property
    @override
    def offset(self) -> Position:
        z_offset = self.model.z_offset if self.model else 0.0
        return Position(self.config.x_offset, self.config.y_offset, self.probe_height + z_offset)

    @property
    @override
    def is_ready(self) -> bool:
        return self.model is not None

    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: ScanModeConfiguration,
        *,
        model: Model | None = None,
        probe_height: float = TRIGGER_DISTANCE,
    ) -> None:
        self._toolhead: Toolhead = toolhead
        self.model: Model | None = model
        self.config: ScanModeConfiguration = config
        self.probe_height: float = probe_height
        self._mcu: Mcu = mcu

    @override
    def perform_probe(self) -> float:
        if not self._toolhead.is_homed("z"):
            msg = "z axis must be homed before probing"
            raise RuntimeError(msg)
        self._toolhead.move(z=self.probe_height, speed=self.config.travel_speed)
        self._toolhead.wait_moves()
        toolhead_pos = self._toolhead.get_position()
        dist = toolhead_pos.z + self.probe_height - self.measure_distance()

        if math.isinf(dist):
            msg = "toolhead stopped outside model range"
            raise RuntimeError(msg)

        pos = self._toolhead.apply_axis_twist_compensation(Position(toolhead_pos.x, toolhead_pos.y, dist))
        logger.info("probe at %.3f,%.3f is z=%.6f", pos.x, pos.y, pos.z)
        return pos.z

    def measure_distance(
        self, *, time: float | None = None, min_sample_count: int | None = None, skip_count: int = 5
    ) -> float:
        min_sample_count = min_sample_count or self.config.samples
        if self.model is None:
            msg = "cannot measure distance without a model"
            raise RuntimeError(msg)
        time = time or self._toolhead.get_last_move_time()

        with self._mcu.start_session(lambda sample: sample.time >= time) as session:
            session.wait_for(lambda samples: len(samples) >= min_sample_count + skip_count)
        samples = session.get_items()[skip_count:]

        dist = float(np.median([self.model.frequency_to_distance(sample.frequency) for sample in samples]))
        return dist

    @override
    def query_is_triggered(self, print_time: float) -> bool:
        distance = self.measure_distance(time=print_time)
        return distance <= self.get_endstop_position()

    @override
    def get_endstop_position(self) -> float:
        return self.probe_height

    @override
    def home_start(self, print_time: float) -> object:
        trigger_frequency = self.get_model().distance_to_frequency(self.probe_height)
        return self._mcu.start_homing_scan(print_time, trigger_frequency)

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        if not homing_state.is_homing_z():
            return
        distance = self.measure_distance()
        if math.isinf(distance):
            msg = "toolhead stopped outside model range"
            raise RuntimeError(msg)

        homing_state.set_z_homed_position(distance)

    @override
    def home_wait(self, home_end_time: float) -> float:
        return self._mcu.stop_homing(home_end_time)

    def start_session(
        self,
    ) -> Session[Sample]:
        time = self._toolhead.get_last_move_time()
        return self._mcu.start_session(lambda sample: sample.time >= time)
