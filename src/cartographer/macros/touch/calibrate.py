from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.configuration import (
    Configuration,
    TouchModelConfiguration,
)
from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.utils import force_home_z
from cartographer.probe.touch_mode import (
    TouchMode,
    TouchModeConfiguration,
    compute_range,
    find_best_subset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.multiprocessing import TaskExecutor
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


logger = logging.getLogger(__name__)


MIN_STEP = 50
MAX_STEP = 1000
DEFAULT_TOUCH_MODEL_NAME = "default"
DEFAULT_Z_OFFSET = -0.05

VERIFICATION_PROBES = 5


@dataclass(frozen=True)
class ScreeningResult:
    """Result from quick screening of a threshold."""

    threshold: int
    samples: tuple[float, ...]
    best_subset: Sequence[float] | None
    best_range: float

    def passed(self, sample_range: float) -> bool:
        """Check if screening found any valid subset."""
        return self.best_range <= sample_range


@dataclass(frozen=True)
class VerificationResult:
    """Result from extended verification of a threshold."""

    threshold: int
    probe_medians: list[float]
    median_range: float

    def passed(self, max_consistency_range: float) -> bool:
        """Check if threshold meets consistency requirements."""
        return self.median_range <= max_consistency_range


def format_distance(distance_mm: float) -> str:
    """
    Format distance with appropriate precision.

    Uses ceiling rounding to ensure non-zero values never display
    as 0.000.
    """
    rounded = ceil(distance_mm * 1000) / 1000
    return f"{rounded:.3f}"


@final
class TouchCalibrateMacro(Macro):
    description = "Run the touch calibration"

    def __init__(
        self,
        probe: Probe,
        mcu: Mcu,
        toolhead: Toolhead,
        config: Configuration,
        task_executor: TaskExecutor,
    ) -> None:
        self._probe = probe
        self._mcu = mcu
        self._toolhead = toolhead
        self._config = config
        self._task_executor = task_executor

    @override
    def run(self, params: MacroParams) -> None:
        name = params.get("MODEL", DEFAULT_TOUCH_MODEL_NAME).lower()
        speed = params.get_int("SPEED", default=2, minval=1, maxval=5)
        threshold_start = params.get_int("START", default=500, minval=100)
        threshold_max = params.get_int(
            "MAX",
            default=5000,
            minval=threshold_start,
        )
        max_consistency_range = params.get_float(
            "MAX_CONSISTENCY_RANGE",
            default=self._config.touch.sample_range * 2,
            minval=self._config.touch.sample_range,
            maxval=self._config.touch.sample_range * 4,
        )
        verification_probes = params.get_int(
            "VERIFICATION_PROBES",
            default=5,
            minval=2,
            maxval=20,
        )

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._move_to_calibration_position()

        required_samples = self._config.touch.samples
        max_samples = self._config.touch.max_samples

        logger.info(
            "Starting touch calibration (speed=%d, range=%d-%d)",
            speed,
            threshold_start,
            threshold_max,
        )
        logger.info(
            "Looking for %d samples within %smm range (max %d attempts per probe, consistency range <= %smm)",
            required_samples,
            format_distance(self._config.touch.sample_range),
            max_samples,
            format_distance(max_consistency_range),
        )

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=threshold_start,
            speed=speed,
        )

        with force_home_z(self._toolhead):
            threshold = self._find_threshold(
                calibration_mode,
                threshold_start,
                threshold_max,
                max_consistency_range,
                verification_probes,
            )

        if threshold is None:
            self._log_calibration_failure(threshold_start, threshold_max)
            return

        self._save_calibration_result(name, threshold, speed)

    def _move_to_calibration_position(self) -> None:
        """Move to the zero reference position for calibration."""
        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

    def _find_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold_start: int,
        threshold_max: int,
        max_consistency_range: float,
        verification_probes: int,
    ) -> int | None:
        """
        Find the minimum threshold that produces consistent results.

        Strategy:
        1. Screen with few samples - pass if any valid subset found
        2. If screening passes, verify with multiple actual touch probes
        3. Accept if probe results are consistent
        """
        threshold = threshold_start
        required_samples = self._config.touch.samples
        screening_samples = ceil(required_samples * 1.5)

        while threshold <= threshold_max:
            # Phase 1: Quick screening
            screening = self._screen_threshold(
                calibration_mode,
                threshold,
                screening_samples,
            )

            if screening is None:
                threshold += self._calculate_step(threshold, None)
                continue

            self._log_screening_result(screening, self._config.touch.sample_range)

            if not screening.passed(self._config.touch.sample_range):
                threshold += self._calculate_step(threshold, screening.best_range)
                continue

            # Phase 2: Actual touch probe verification
            verification = self._verify_threshold(
                calibration_mode,
                threshold,
                max_consistency_range,
                verification_probes,
            )

            if verification is None:
                threshold += self._calculate_step(threshold, None)
                continue

            if verification.passed(max_consistency_range):
                logger.info(
                    "Threshold %d verified: %smm median range across %d probes",
                    threshold,
                    format_distance(verification.median_range),
                    len(verification.probe_medians),
                )
                return threshold

            # Consistency check failed - increase threshold
            logger.debug(
                "Verification failed: median range %smm > %smm, increasing threshold",
                format_distance(verification.median_range),
                format_distance(max_consistency_range),
            )
            threshold += self._calculate_step(threshold, verification.median_range)

        return None

    def _screen_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold: int,
        sample_count: int,
    ) -> ScreeningResult | None:
        """
        Quick screen: can we find any valid subset?

        Returns None if probe triggered due to noise.
        """
        try:
            samples = calibration_mode.collect_samples(threshold, sample_count)
        except ProbeTriggerError:
            logger.warning(
                "Threshold %d triggered prior to movement.",
                threshold,
            )
            return None

        required = self._config.touch.samples
        best = find_best_subset(samples, required)
        best_range = compute_range(best) if best else float("inf")

        return ScreeningResult(
            threshold=threshold,
            samples=samples,
            best_subset=best,
            best_range=best_range,
        )

    def _verify_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold: int,
        max_consistency_range: float,
        verification_probes: int,
    ) -> VerificationResult | None:
        """
        Verify threshold by running actual touch probe sequences.

        Performs multiple complete touch probe attempts and checks that
        the resulting medians are consistent. Exits early if the median
        range already exceeds the consistency limit.

        Returns None if probe triggered due to noise.
        """
        logger.info(
            "Threshold %d looks promising, verifying with %d touch probes...",
            threshold,
            verification_probes,
        )

        calibration_mode.set_threshold(threshold)
        probe_medians: list[float] = []

        for attempt in range(verification_probes):
            try:
                median = calibration_mode.perform_touch_probe()
                probe_medians.append(median)
                logger.debug(
                    "Verification probe %d/%d: median=%.4fmm",
                    attempt + 1,
                    verification_probes,
                    median,
                )
            except ProbeTriggerError:
                logger.warning(
                    "Threshold %d triggered prior to movement on probe %d.",
                    threshold,
                    attempt + 1,
                )
                return None

            # Early exit: no point continuing if already inconsistent
            if len(probe_medians) >= 2:
                current_range = float(np.max(probe_medians) - np.min(probe_medians))
                if current_range > max_consistency_range:
                    logger.debug(
                        "Early exit: median range %smm > %smm after %d probes",
                        format_distance(current_range),
                        format_distance(max_consistency_range),
                        len(probe_medians),
                    )
                    break

        median_range = float(np.max(probe_medians) - np.min(probe_medians))

        result = VerificationResult(
            threshold=threshold,
            probe_medians=probe_medians,
            median_range=median_range,
        )

        self._log_verification_result(result, max_consistency_range)
        return result

    def _calculate_step(self, threshold: int, range_value: float | None) -> int:
        """
        Calculate step size based on how far from target we are.

        Larger steps when range is very bad, smaller steps when close.
        """
        if range_value is None:
            return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.20)))
        if range_value > self._config.touch.sample_range * 10:
            return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.20)))
        return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.10)))

    def _log_calibration_failure(
        self,
        threshold_start: int,
        threshold_max: int,
    ) -> None:
        """Log failure message with suggested next steps."""
        logger.info(
            "Failed to find reliable threshold in range %d-%d.\n"
            "Try increasing MAX:\n"
            "CARTOGRAPHER_TOUCH_CALIBRATE START=%d MAX=%d",
            threshold_start,
            threshold_max,
            threshold_max,
            int(threshold_max * 1.5),
        )

    def _save_calibration_result(
        self,
        name: str,
        threshold: int,
        speed: int,
    ) -> None:
        """Save the calibration result and log success."""
        logger.info(
            "Calibration complete: threshold=%d, speed=%d",
            threshold,
            speed,
        )
        model = TouchModelConfiguration(name=name, threshold=threshold, speed=speed, z_offset=DEFAULT_Z_OFFSET)
        self._config.save_touch_model(model)
        self._probe.touch.load_model(name)
        logger.info(
            "Touch model '%s' has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config "
            "file and restart the printer.",
            name,
        )

    def _log_screening_result(
        self,
        result: ScreeningResult,
        sample_range: float,
    ) -> None:
        """Log a screening result."""
        status = "✓" if result.passed(sample_range) else "✗"
        logger.info(
            "Screening %d: %s best=%smm (%d samples)",
            result.threshold,
            status,
            format_distance(result.best_range),
            len(result.samples),
        )

        if logger.isEnabledFor(logging.DEBUG):
            samples_str = ", ".join(f"{s:.4f}" for s in result.samples)
            best_str = ", ".join(f"{s:.4f}" for s in result.best_subset) if result.best_subset else "none"
            logger.debug(
                "Screening %d details:\n  samples: [%s]\n  best subset: [%s]\n  best range: %s mm",
                result.threshold,
                samples_str,
                best_str,
                format_distance(result.best_range),
            )

    def _log_verification_result(
        self,
        result: VerificationResult,
        max_consistency_range: float,
    ) -> None:
        """Log a verification result."""
        status = "✓" if result.passed(max_consistency_range) else "✗"
        logger.info(
            "Verification %d: %s median_range=%smm (%d probes)",
            result.threshold,
            status,
            format_distance(result.median_range),
            len(result.probe_medians),
        )

        if logger.isEnabledFor(logging.DEBUG):
            medians_str = ", ".join(f"{m:.4f}" for m in result.probe_medians)
            logger.debug(
                "Verification %d details:\n"
                "  probe medians: [%s]\n"
                "  median range: %s mm\n"
                "  max consistency range: %s mm",
                result.threshold,
                medians_str,
                format_distance(result.median_range),
                format_distance(max_consistency_range),
            )


@final
class CalibrationTouchMode(TouchMode):
    """Touch mode configured for calibration."""

    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: TouchModeConfiguration,
        *,
        threshold: int,
        speed: float,
    ) -> None:
        model = TouchModelConfiguration(name="calibration", threshold=threshold, speed=speed, z_offset=0)
        super().__init__(
            mcu,
            toolhead,
            replace(config, models={"calibration": model}),
        )
        self.load_model("calibration")

    def set_threshold(self, threshold: int) -> None:
        """Update the calibration threshold."""
        self._models["calibration"] = replace(
            self._models["calibration"],
            threshold=threshold,
        )
        self.load_model("calibration")

    def collect_samples(
        self,
        threshold: int,
        sample_count: int,
    ) -> tuple[float, ...]:
        """Collect samples at the given threshold."""
        self.set_threshold(threshold)
        samples: list[float] = []

        for _ in range(sample_count):
            pos = self._perform_single_probe()
            samples.append(pos)

        return tuple(sorted(samples))

    def perform_touch_probe(self) -> float:
        """
        Perform one complete touch probe sequence.

        This simulates exactly what happens at runtime: collect up to
        max_samples, find the best subset, return the median.
        """
        return self._run_probe()
