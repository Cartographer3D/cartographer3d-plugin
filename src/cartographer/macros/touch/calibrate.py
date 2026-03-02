from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from math import ceil, inf, isfinite
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import Protocol, override, runtime_checkable

from cartographer.interfaces.configuration import (
    Configuration,
    TouchModelConfiguration,
)
from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.utils import force_home_z
from cartographer.probe.touch_mode import (
    TouchError,
    TouchMode,
    TouchModeConfiguration,
    compute_range,
    find_best_subset,
    run_probe_sequence,
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

    def passed(self, max_verify_range: float) -> bool:
        """Check if threshold meets consistency requirements."""
        return self.median_range <= max_verify_range


def format_distance(distance_mm: float) -> str:
    """
    Format distance with appropriate precision.

    Uses ceiling rounding to ensure non-zero values never display
    as 0.000.
    """
    if not isfinite(distance_mm):
        return "inf" if distance_mm == inf else str(distance_mm)
    rounded = ceil(distance_mm * 1000) / 1000
    return f"{rounded:.3f}"


def calculate_step(threshold: int, range_value: float | None, sample_range: float) -> int:
    """
    Calculate step size based on how far from target we are.

    Larger steps when range is very bad, smaller steps when close.
    """
    if range_value is None:
        return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.20)))
    if range_value > sample_range * 10:
        return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.20)))
    return min(MAX_STEP, max(MIN_STEP, int(threshold * 0.10)))


@runtime_checkable
class CalibrationProbe(Protocol):
    """Protocol for probe operations needed during calibration."""

    def collect_samples(self, threshold: int, sample_count: int) -> tuple[float, ...]:
        """Collect raw samples at the given threshold."""
        ...

    def set_threshold(self, threshold: int) -> None:
        """Update the active threshold."""
        ...

    def perform_touch_probe(self) -> float:
        """Perform one complete touch probe sequence."""
        ...


@final
class ThresholdScreener:
    """Quick-screens a threshold by collecting samples and finding a valid subset."""

    def __init__(self, probe: CalibrationProbe, required_samples: int) -> None:
        self._probe = probe
        self._required_samples = required_samples

    def screen(self, threshold: int, sample_count: int) -> ScreeningResult | None:
        """
        Quick screen: can we find any valid subset?

        Returns None if probe triggered due to noise.
        """
        try:
            samples = self._probe.collect_samples(threshold, sample_count)
        except ProbeTriggerError:
            logger.warning(
                "Threshold %d triggered prior to movement.",
                threshold,
            )
            return None

        best = find_best_subset(samples, self._required_samples)
        best_range = compute_range(best) if best else float("inf")

        return ScreeningResult(
            threshold=threshold,
            samples=samples,
            best_subset=best,
            best_range=best_range,
        )


@final
class ThresholdVerifier:
    """Verifies a threshold by running multiple full probe sequences."""

    def __init__(self, probe: CalibrationProbe) -> None:
        self._probe = probe

    def verify(
        self,
        threshold: int,
        max_verify_range: float,
        sample_count: int,
    ) -> VerificationResult | None:
        """
        Verify threshold by running actual touch probe sequences.

        Performs multiple complete touch probe attempts and checks that
        the resulting medians are consistent. Exits early if the median
        range already exceeds the limit.

        Returns None if probe triggered due to noise.
        """
        logger.info(
            "Threshold %d looks promising, verifying with %d touch samples...",
            threshold,
            sample_count,
        )

        self._probe.set_threshold(threshold)
        medians: list[float] = []

        for attempt in range(sample_count):
            try:
                median = self._probe.perform_touch_probe()
                medians.append(median)
                logger.debug(
                    "Verification sample %d/%d: median=%.4fmm",
                    attempt + 1,
                    sample_count,
                    median,
                )
            except ProbeTriggerError:
                logger.warning(
                    "Threshold %d triggered prior to movement on sample %d.",
                    threshold,
                    attempt + 1,
                )
                return None
            except TouchError:
                logger.warning(
                    "Threshold %d failed to find consistent samples on sample %d.",
                    threshold,
                    attempt + 1,
                )
                return None

            # Early exit: no point continuing if already inconsistent
            if len(medians) >= 2:
                current_range = float(np.max(medians) - np.min(medians))
                if current_range > max_verify_range:
                    logger.debug(
                        "Early exit: median range %smm > %smm after %d samples",
                        format_distance(current_range),
                        format_distance(max_verify_range),
                        len(medians),
                    )
                    break

        median_range = float(np.max(medians) - np.min(medians))

        return VerificationResult(
            threshold=threshold,
            probe_medians=medians,
            median_range=median_range,
        )


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
        max_verify_range = params.get_float(
            "MAX_VERIFY_RANGE",
            default=self._config.touch.sample_range * 2,
            minval=self._config.touch.sample_range,
            maxval=self._config.touch.sample_range * 4,
        )
        verification_samples = params.get_int(
            "VERIFICATION_SAMPLES",
            default=10,
            minval=3,
            maxval=20,
        )

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._move_to_calibration_position()

        required_samples = self._config.touch.samples
        sample_range = self._config.touch.sample_range

        logger.info(
            "Starting touch calibration (speed=%d, range=%d-%d)",
            speed,
            threshold_start,
            threshold_max,
        )
        logger.info(
            "Screening: %d samples within %smm range. Verification: %d samples within %smm range",
            required_samples,
            format_distance(sample_range),
            verification_samples,
            format_distance(max_verify_range),
        )

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=threshold_start,
            speed=speed,
        )

        screener = ThresholdScreener(calibration_mode, required_samples)
        verifier = ThresholdVerifier(calibration_mode)

        with force_home_z(self._toolhead):
            threshold = self._find_threshold(
                screener,
                verifier,
                threshold_start,
                threshold_max,
                sample_range,
                max_verify_range,
                verification_samples,
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
        screener: ThresholdScreener,
        verifier: ThresholdVerifier,
        threshold_start: int,
        threshold_max: int,
        sample_range: float,
        max_verify_range: float,
        verification_samples: int,
    ) -> int | None:
        """
        Find the minimum threshold that produces consistent results.

        Strategy:
        1. Screen with few samples - pass if any valid subset found
        2. If screening passes, verify with multiple actual touch probes
        3. Accept if probe results are consistent
        """
        threshold = threshold_start
        screening_samples = self._config.touch.samples + self._config.touch.max_noisy_samples

        while threshold <= threshold_max:
            # Phase 1: Quick screening
            screening = screener.screen(threshold, screening_samples)

            if screening is None:
                threshold += calculate_step(threshold, None, sample_range)
                continue

            self._log_screening_result(screening, sample_range)

            if not screening.passed(sample_range):
                threshold += calculate_step(threshold, screening.best_range, sample_range)
                continue

            # Phase 2: Actual touch probe verification
            verification = verifier.verify(threshold, max_verify_range, verification_samples)

            if verification is None:
                threshold += calculate_step(threshold, None, sample_range)
                continue

            self._log_verification_result(verification, max_verify_range)

            if verification.passed(max_verify_range):
                logger.info(
                    "Threshold %d verified: %smm median range across %d samples",
                    threshold,
                    format_distance(verification.median_range),
                    len(verification.probe_medians),
                )
                return threshold

            # Consistency check failed - increase threshold
            logger.debug(
                "Verification failed: median range %smm > %smm, increasing threshold",
                format_distance(verification.median_range),
                format_distance(max_verify_range),
            )
            threshold += calculate_step(threshold, verification.median_range, sample_range)

        return None

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
        max_verify_range: float,
    ) -> None:
        """Log a verification result with touch-accuracy-style report."""
        status = "✓" if result.passed(max_verify_range) else "✗"
        medians = result.probe_medians

        max_value = max(medians)
        min_value = min(medians)
        range_value = max_value - min_value
        avg_value = float(np.mean(medians))
        median_value = float(np.median(medians))
        std_dev = float(np.std(medians))

        logger.info(
            "Verification %d: %s (%d samples)\n"
            "  maximum %.6f, minimum %.6f, range %.6f,\n"
            "  average %.6f, median %.6f,\n"
            "  standard deviation %.6f",
            result.threshold,
            status,
            len(medians),
            max_value,
            min_value,
            range_value,
            avg_value,
            median_value,
            std_dev,
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

        Uses the shared run_probe_sequence to ensure parity with
        the runtime TouchMode._run_probe() logic.
        """
        return run_probe_sequence(
            self._perform_single_probe,
            samples=self._config.samples,
            max_samples=self._config.max_samples,
            max_window=self._config.max_window,
            sample_range=self._config.sample_range,
        )
