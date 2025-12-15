from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.configuration import (
    Configuration,
    TouchModelConfiguration,
)
from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.utils import force_home_z
from cartographer.probe.touch_mode import (
    MAX_SAMPLE_RANGE,
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

DEFAULT_SUCCESS_RATE = 0.95
SIMULATION_COUNT = 1000


@dataclass(frozen=True)
class ScreeningResult:
    """Result from quick screening of a threshold."""

    threshold: int
    samples: tuple[float, ...]
    best_subset: Sequence[float] | None
    best_range: float

    @property
    def passed(self) -> bool:
        """Check if screening found any valid subset."""
        return self.best_range <= MAX_SAMPLE_RANGE


@dataclass(frozen=True)
class VerificationResult:
    """Result from extended verification of a threshold."""

    threshold: int
    samples: tuple[float, ...]
    best_subset: Sequence[float] | None
    best_range: float
    success_rate: float

    def passed(self, required_rate: float) -> bool:
        """Check if the threshold meets the required success rate."""
        return self.success_rate >= required_rate


def would_probing_pass(
    samples: Sequence[float],
    required_samples: int,
) -> bool:
    """
    Check if a set of samples would pass at runtime.

    Simulates _run_probe logic: pass if any subset of required_samples
    has range <= MAX_SAMPLE_RANGE.
    """
    if len(samples) < required_samples:
        return False
    sorted_samples = sorted(samples)
    for i in range(len(sorted_samples) - required_samples + 1):
        window = sorted_samples[i : i + required_samples]
        if window[-1] - window[0] <= MAX_SAMPLE_RANGE:
            return True
    return False


def estimate_success_rate(
    samples: Sequence[float],
    max_probed_samples: int,
    required_samples: int,
    simulation_count: int = SIMULATION_COUNT,
) -> float:
    """
    Estimate success rate by Monte Carlo simulation.

    Simulates runtime behavior by randomly selecting max_probed_samples
    samples from the pool, then checking if that set would pass
    (i.e., contains at least one valid required_samples subset).

    Parameters
    ----------
    samples
        The pool of collected samples to draw from.
    max_probed_samples
        Number of samples collected during a probe sequence.
    required_samples
        Number of samples needed to form a valid subset.
    simulation_count
        Number of Monte Carlo simulations to run.

    Returns
    -------
        Estimated probability that a runtime probe sequence will succeed.
    """
    if len(samples) < max_probed_samples:
        return 0.0

    samples_list = list(samples)
    passing = 0

    for _ in range(simulation_count):
        runtime_set = random.sample(samples_list, max_probed_samples)
        if would_probing_pass(runtime_set, required_samples):
            passing += 1

    return passing / simulation_count


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
        required_success_rate = params.get_float(
            "SUCCESS_RATE",
            default=DEFAULT_SUCCESS_RATE,
            minval=0.1,
            maxval=0.99,
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
            "Looking for %d samples within %.3fmm range (max %d attempts, %.0f%% success rate required)",
            required_samples,
            MAX_SAMPLE_RANGE,
            max_samples,
            required_success_rate * 100,
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
                required_success_rate,
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
        required_success_rate: float,
    ) -> int | None:
        """
        Find the minimum threshold that produces consistent results.

        Strategy:
        1. Screen with few samples - pass if any valid subset found
        2. If screening passes, verify with many samples
        3. Require high success rate for verification to pass
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

            self._log_screening_result(screening)

            if not screening.passed:
                threshold += self._calculate_step(threshold, screening.best_range)
                continue

            # Phase 2: Extended verification
            verification = self._verify_threshold(
                calibration_mode,
                threshold,
                required_success_rate,
            )

            if verification is None:
                threshold += self._calculate_step(threshold, None)
                continue

            if verification.passed(required_success_rate):
                logger.info(
                    "Threshold %d verified: %.0f%% success rate",
                    threshold,
                    verification.success_rate * 100,
                )
                return threshold

            logger.debug(
                "Verification failed (%.0f%% < %.0f%%), stepping up",
                verification.success_rate * 100,
                required_success_rate * 100,
            )
            threshold += MIN_STEP

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
        required_success_rate: float,
    ) -> VerificationResult | None:
        """
        Extended verification with success rate estimation.

        Uses max_samples from config for runtime simulation and
        collects 2.5x max_samples for statistical confidence.

        Returns None if probe triggered due to noise.
        """
        logger.info("Threshold %d looks promising, verifying...", threshold)

        required_samples = self._config.touch.samples
        max_samples = self._config.touch.max_samples
        verification_samples = ceil(max_samples * 2.5)

        try:
            samples = calibration_mode.collect_samples(
                threshold,
                verification_samples,
            )
        except ProbeTriggerError:
            logger.warning(
                "Threshold %d triggered prior to movement.",
                threshold,
            )
            return None

        best = find_best_subset(samples, required_samples)
        best_range = compute_range(best) if best else float("inf")

        success_rate = self._task_executor.run(
            estimate_success_rate,
            samples,
            max_probed_samples=max_samples,
            required_samples=required_samples,
        )

        result = VerificationResult(
            threshold=threshold,
            samples=samples,
            best_subset=best,
            best_range=best_range,
            success_rate=success_rate,
        )

        self._log_verification_result(result, required_success_rate)
        return result

    def _calculate_step(self, threshold: int, range_value: float | None) -> int:
        """
        Calculate step size based on how far from target we are.

        Larger steps when range is very bad, smaller steps when close.
        """
        if range_value is None:
            return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.20)))
        if range_value > MAX_SAMPLE_RANGE * 10:
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
        model = TouchModelConfiguration(name, threshold, speed, DEFAULT_Z_OFFSET)
        self._config.save_touch_model(model)
        self._probe.touch.load_model(name)
        logger.info(
            "Touch model '%s' has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config "
            "file and restart the printer.",
            name,
        )

    def _log_screening_result(self, result: ScreeningResult) -> None:
        """Log a screening result."""
        status = "✓" if result.passed else "✗"
        logger.info(
            "Screening %d: %s best=%.4fmm (%d samples)",
            result.threshold,
            status,
            result.best_range,
            len(result.samples),
        )

        if logger.isEnabledFor(logging.DEBUG):
            samples_str = ", ".join(f"{s:.4f}" for s in result.samples)
            best_str = ", ".join(f"{s:.4f}" for s in result.best_subset) if result.best_subset else "none"
            logger.debug(
                "Screening %d details:\n  samples: [%s]\n  best subset: [%s]\n  best range: %.4f mm",
                result.threshold,
                samples_str,
                best_str,
                result.best_range,
            )

    def _log_verification_result(
        self,
        result: VerificationResult,
        required_success_rate: float,
    ) -> None:
        """Log a verification result."""
        status = "✓" if result.passed(required_success_rate) else "✗"
        logger.info(
            "Verification %d: %s best=%.4fmm, success_rate=%.0f%% (%d samples)",
            result.threshold,
            status,
            result.best_range,
            result.success_rate * 100,
            len(result.samples),
        )

        if logger.isEnabledFor(logging.DEBUG):
            samples_str = ", ".join(f"{s:.4f}" for s in result.samples)
            best_str = ", ".join(f"{s:.4f}" for s in result.best_subset) if result.best_subset else "none"
            logger.debug(
                "Verification %d details:\n"
                "  samples: [%s]\n"
                "  best subset: [%s]\n"
                "  best range: %.4f mm\n"
                "  success rate: %.1f%%\n"
                "  required rate: %.1f%%",
                result.threshold,
                samples_str,
                best_str,
                result.best_range,
                result.success_rate * 100,
                required_success_rate * 100,
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
        model = TouchModelConfiguration("calibration", threshold, speed, 0)
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
