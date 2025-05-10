from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.configuration import Configuration, TouchModelConfiguration
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.lib.statistics import compute_mad
from cartographer.macros.utils import get_choice
from cartographer.probe.touch_mode import MAD_TOLERANCE, TouchMode, TouchModeConfiguration

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


logger = logging.getLogger(__name__)


MIN_ALLOWED_STEP = 75
MAX_ALLOWED_STEP = 500


class CalibrationStrategy(ABC):
    @abstractmethod
    def compute_score(self, samples: list[float]) -> float: ...
    @abstractmethod
    def is_acceptable(self, score: float) -> bool: ...
    @abstractmethod
    def should_stop_early(self, samples: list[float]) -> bool: ...
    @abstractmethod
    def compute_step_increase(self, current_threshold: int, score: float) -> int: ...


@final
class DefaultCalibrationStrategy(CalibrationStrategy):
    """Standard MAD-based calibration strategy."""

    @override
    def compute_score(self, samples: list[float]) -> float:
        return compute_mad(samples)

    @override
    def is_acceptable(self, score: float) -> bool:
        return score <= MAD_TOLERANCE

    @override
    def should_stop_early(self, samples: list[float]) -> bool:
        if len(samples) >= 3:
            return False
        return self.compute_score(samples) > 5 * MAD_TOLERANCE

    @override
    def compute_step_increase(self, current_threshold: int, score: float) -> int:
        allowed_step = (current_threshold * 0.06) ** 1.15
        allowed_max_step = max(MIN_ALLOWED_STEP, min(MAX_ALLOWED_STEP, allowed_step))
        ratio = max(1.0, score / MAD_TOLERANCE)
        raw_step = allowed_max_step * (1 - (1 / ratio))
        return int(round(min(allowed_max_step, max(MIN_ALLOWED_STEP, raw_step))))


@final
class AggressiveCalibrationStrategy(CalibrationStrategy):
    """Faster calibration with larger threshold steps."""

    @override
    def compute_score(self, samples: list[float]) -> float:
        return compute_mad(samples)

    @override
    def is_acceptable(self, score: float) -> bool:
        return score <= MAD_TOLERANCE * 1.2  # Slightly more lenient

    @override
    def should_stop_early(self, samples: list[float]) -> bool:
        if len(samples) >= 3:
            return False
        return self.compute_score(samples) > 10 * MAD_TOLERANCE  # Only stop for extreme cases

    @override
    def compute_step_increase(self, current_threshold: int, score: float) -> int:
        # Larger steps to find threshold faster
        base_step = min(300, current_threshold // 3)
        ratio = max(1.0, score / MAD_TOLERANCE)
        return int(round(base_step * ratio))


@final
class PrecisionCalibrationStrategy(CalibrationStrategy):
    """Slower but more precise calibration."""

    @override
    def compute_score(self, samples: list[float]) -> float:
        # Use both MAD and range for stricter evaluation
        mad = compute_mad(samples)
        sample_range = max(samples) - min(samples)
        return max(mad, sample_range / 2)

    @override
    def is_acceptable(self, score: float) -> bool:
        return score <= MAD_TOLERANCE * 0.8  # Stricter tolerance

    @override
    def should_stop_early(self, samples: list[float]) -> bool:
        if len(samples) >= 3:
            return False
        return self.compute_score(samples) > 3 * MAD_TOLERANCE

    @override
    def compute_step_increase(self, current_threshold: int, score: float) -> int:
        allowed_step = (current_threshold * 0.06) ** 1.15
        allowed_max_step = max(MIN_ALLOWED_STEP, min(MAX_ALLOWED_STEP, allowed_step))
        ratio = max(1.0, score / MAD_TOLERANCE)
        raw_step = allowed_max_step * (1 - (1 / ratio))
        return int(round(min(allowed_max_step, max(MIN_ALLOWED_STEP, raw_step))))


@final
class RobustCalibrationStrategy(CalibrationStrategy):
    @override
    def compute_score(self, samples: list[float]) -> float:
        # Ensure we have enough samples
        if len(samples) <= 3:
            return float("inf")

        sorted_samples = sorted(samples)

        # Core metrics (tuned for 5 samples)
        metrics = {
            "mad": 0.50,  # Primary stability indicator
            "iqr": 0.30,  # Spread of middle 50%
            "range_90": 0.15,  # Middle 90% range (ignores extreme 5% each side)
            "drift": 0.05,  # Measurement drift over time
        }

        # Calculate each component
        mad = compute_mad(samples)
        iqr = sorted_samples[3] - sorted_samples[1]  # Simulated IQR for 5 samples
        range_90 = sorted_samples[-2] - sorted_samples[1]  # Ignore single worst on each end
        drift = abs(np.mean(samples[:2]) - np.mean(samples[-2:]))  # End-to-end drift

        # Weighted composite score
        return float(
            metrics["mad"] * mad + metrics["iqr"] * iqr + metrics["range_90"] * range_90 + metrics["drift"] * abs(drift)
        )

    @override
    def is_acceptable(self, score: float) -> bool:
        """Strict but not overly sensitive criteria"""
        return score <= MAD_TOLERANCE * 1.5

    @override
    def should_stop_early(self, samples: list[float]) -> bool:
        """Only stop for clearly unstable thresholds"""
        if len(samples) >= 3:
            return False
        return self.compute_score(samples) > MAD_TOLERANCE * 5

    @override
    def compute_step_increase(self, current_threshold: int, score: float) -> int:
        allowed_step = (current_threshold * 0.06) ** 1.15
        allowed_max_step = max(MIN_ALLOWED_STEP, min(MAX_ALLOWED_STEP, allowed_step))
        ratio = max(1.0, score / MAD_TOLERANCE)
        raw_step = allowed_max_step * (1 - (1 / ratio))
        return int(round(min(allowed_max_step, max(MIN_ALLOWED_STEP, raw_step))))


STRATEGY_MAP = {
    "default": DefaultCalibrationStrategy,
    "aggressive": AggressiveCalibrationStrategy,
    "precision": PrecisionCalibrationStrategy,
    "robust": RobustCalibrationStrategy,
}


@final
class TouchCalibrateMacro(Macro):
    name = "TOUCH_CALIBRATE"
    description = "Run the touch calibration"

    def __init__(self, probe: Probe, mcu: Mcu, toolhead: Toolhead, config: Configuration) -> None:
        self._probe = probe
        self._mcu = mcu
        self._toolhead = toolhead
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        name = params.get("MODEL_NAME", "default")
        speed = params.get_int("SPEED", default=3, minval=1, maxval=5)
        threshold_start = params.get_int("START", default=500, minval=100)
        threshold_max = params.get_int("MAX", default=3000, minval=threshold_start)
        strategy_type = get_choice(params, "STRATEGY", default="default", choices=STRATEGY_MAP.keys())
        strategy = STRATEGY_MAP[strategy_type]()

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "must home x and y before calibration"
            raise RuntimeError(msg)

        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

        logger.info(
            "Starting %s touch calibration at speed %d, threshold %d-%d",
            strategy_type,
            speed,
            threshold_start,
            threshold_max,
        )

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=threshold_start,
            speed=speed,
            strategy=strategy,
        )

        forced_z = False
        try:
            if not self._toolhead.is_homed("z"):
                forced_z = True
                _, z_max = self._toolhead.get_z_axis_limits()
                self._toolhead.set_z_position(z=z_max - 10)

            threshold = self._find_acceptable_threshold(calibration_mode, threshold_start, threshold_max)
        finally:
            if forced_z:
                self._toolhead.clear_z_homing_state()

        if threshold is None:
            logger.info(
                """
                Failed to calibrate with %s strategy (thresholds %d-%d).
                Try increasing MAX or changing strategy.
                TOUCH_CALIBRATE START=%d MAX=%d
                """,
                strategy_type,
                threshold_start,
                threshold_max,
                threshold_max,
                threshold_max + 2000,
            )
            return

        logger.info("Successfully calibrated with %s strategy (threshold %d)", strategy_type, threshold)
        model = TouchModelConfiguration(name, threshold, speed, 0)
        self._config.save_touch_model(model)
        logger.info("Saved touch model '%s'", name)

    def _find_acceptable_threshold(
        self, calibration_mode: CalibrationTouchMode, threshold_start: int, threshold_max: int
    ) -> int | None:
        current_threshold = threshold_start
        strategy = calibration_mode.strategy

        while current_threshold <= threshold_max:
            samples = calibration_mode.collect_samples(current_threshold)
            score = strategy.compute_score(samples)

            logger.info(
                "Threshold %d score: %.6f",
                current_threshold,
                score,
            )
            logger.debug("Samples: %s", ", ".join(f"{s:.6f}" for s in samples))

            if strategy.is_acceptable(score):
                logger.info("Threshold %d accepted (score %.6f)", current_threshold, score)
                return current_threshold

            step = strategy.compute_step_increase(current_threshold, score)
            current_threshold += step
            logger.info("Next threshold: %d (+%d)", current_threshold, step)

        return None


@final
class CalibrationTouchMode(TouchMode):
    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: TouchModeConfiguration,
        *,
        threshold: int,
        speed: float,
        strategy: CalibrationStrategy,
    ) -> None:
        model = TouchModelConfiguration("calibration", threshold, speed, 0)
        super().__init__(mcu, toolhead, replace(config, models={"calibration": model}))
        self.load_model("calibration")
        self.strategy = strategy

    def set_threshold(self, threshold: int) -> None:
        self._models["calibration"] = replace(self._models["calibration"], threshold=threshold)
        self.load_model("calibration")

    def collect_samples(self, threshold: int) -> list[float]:
        samples: list[float] = []
        self.set_threshold(threshold)

        for _ in range(self._config.samples * 3):
            pos = self._perform_single_probe()
            samples.append(pos)

            if self.strategy.should_stop_early(samples):
                break

        return sorted(samples)
