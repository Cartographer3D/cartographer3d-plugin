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
from cartographer.macros.fields import param, parse
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

    def perform_touch_probe(self, *, z_limit: float | None = None) -> float:
        """Perform one complete touch probe sequence."""
        ...


@runtime_checkable
class Screener(Protocol):
    """Protocol for threshold screening operations."""

    def screen(self, threshold: int, sample_count: int) -> ScreeningResult | None:
        """Quick screen a threshold. Returns None on probe trigger error."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for threshold verification operations."""

    def verify(
        self,
        threshold: int,
        max_verify_range: float,
        sample_count: int,
        *,
        z_limit: float | None = None,
    ) -> VerificationResult | None:
        """Verify a threshold with multiple touch probes. Returns None on failure."""
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
        *,
        z_limit: float | None = None,
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
                median = self._probe.perform_touch_probe(z_limit=z_limit)
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


@dataclass(frozen=True)
class TouchCalibrateParams:
    """Parameters for CARTOGRAPHER_TOUCH_CALIBRATE."""

    max_verify_range: float = param("Maximum verification range")
    model: str = param("Model name", default=DEFAULT_TOUCH_MODEL_NAME)
    speed: int = param("Probing speed", default=2, min=1, max=5)
    threshold_start: int = param("Starting threshold", default=500, min=100, key="START")
    threshold_max: int = param("Maximum threshold", default=5000, min=100, key="MAX")
    verification_samples: int = param("Verification sample count", default=10, min=3, max=20)


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
        sample_range = self._config.touch.sample_range
        p = parse(
            TouchCalibrateParams,
            params,
            max_verify_range=self._config.touch.sample_range * 2,
        )
        name = p.model.lower()

        if p.threshold_max < p.threshold_start:
            msg = f"MAX ({p.threshold_max}) must be >= START ({p.threshold_start})"
            raise RuntimeError(msg)
        if p.max_verify_range < sample_range:
            msg = f"MAX_VERIFY_RANGE ({p.max_verify_range}) must be >= sample_range ({sample_range})"
            raise RuntimeError(msg)
        if p.max_verify_range > sample_range * 4:
            msg = f"MAX_VERIFY_RANGE ({p.max_verify_range}) must be <= sample_range * 4 ({sample_range * 4})"
            raise RuntimeError(msg)

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._move_to_calibration_position()

        required_samples = self._config.touch.samples

        logger.info(
            "Starting touch calibration (speed=%d, range=%d-%d)",
            p.speed,
            p.threshold_start,
            p.threshold_max,
        )
        logger.info(
            "Screening: %d samples within %smm range. Verification: %d samples within %smm range",
            required_samples,
            format_distance(sample_range),
            p.verification_samples,
            format_distance(p.max_verify_range),
        )

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=p.threshold_start,
            speed=p.speed,
        )

        screener = ThresholdScreener(calibration_mode, required_samples)
        verifier = ThresholdVerifier(calibration_mode)

        with force_home_z(self._toolhead):
            threshold = self._find_threshold(
                screener,
                verifier,
                p.threshold_start,
                p.threshold_max,
                sample_range,
                p.max_verify_range,
                p.verification_samples,
            )

        if threshold is None:
            self._log_calibration_failure(p.threshold_start, p.threshold_max)
            return

        self._save_calibration_result(name, threshold, p.speed)

    def _move_to_calibration_position(self) -> None:
        """Move to the zero reference position for calibration."""
        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

    def _try_threshold(
        self,
        screener: Screener,
        verifier: Verifier,
        threshold: int,
        screening_samples: int,
        sample_range: float,
        max_verify_range: float,
        verification_samples: int,
        *,
        z_limit: float | None = None,
    ) -> tuple[VerificationResult | None, int]:
        """
        Screen and verify a single threshold candidate.

        Returns (result, step) where result is the VerificationResult
        if both screening and verification succeeded, or None if either
        phase failed. step is the recommended threshold increment.
        """
        screening = screener.screen(threshold, screening_samples)

        if screening is None:
            return None, calculate_step(threshold, None, sample_range)

        self._log_screening_result(screening, sample_range)

        if not screening.passed(sample_range):
            return None, calculate_step(threshold, screening.best_range, sample_range)

        verification = verifier.verify(threshold, max_verify_range, verification_samples, z_limit=z_limit)

        if verification is None:
            return None, calculate_step(threshold, None, sample_range)

        self._log_verification_result(verification, max_verify_range)
        return verification, calculate_step(threshold, verification.median_range, sample_range)

    def _find_threshold(
        self,
        screener: Screener,
        verifier: Verifier,
        threshold_start: int,
        threshold_max: int,
        sample_range: float,
        max_verify_range: float,
        verification_samples: int,
    ) -> int | None:
        """
        Find the best threshold that produces consistent results.

        Strategy:
        1. Screen with few samples - pass if any valid subset found
        2. If screening passes, verify with multiple actual touch probes
        3. Once a passing threshold is found, optimize by searching higher
           thresholds (up to 20% above) with a z_limit safety bound
        4. Return the threshold with the best (lowest) median range
        """
        threshold = threshold_start
        screening_samples = self._config.touch.samples + self._config.touch.max_noisy_samples

        while threshold <= threshold_max:
            verification, step = self._try_threshold(
                screener,
                verifier,
                threshold,
                screening_samples,
                sample_range,
                max_verify_range,
                verification_samples,
            )

            if verification is None:
                threshold += step
                continue

            if verification.passed(max_verify_range):
                logger.info(
                    "Threshold %d verified: %smm median range across %d samples",
                    threshold,
                    format_distance(verification.median_range),
                    len(verification.probe_medians),
                )
                return self._optimize_threshold(
                    screener,
                    verifier,
                    threshold,
                    verification,
                    threshold_max,
                    sample_range,
                    max_verify_range,
                    verification_samples,
                )

            # Consistency check failed - increase threshold
            logger.debug(
                "Verification failed: median range %smm > %smm, increasing threshold",
                format_distance(verification.median_range),
                format_distance(max_verify_range),
            )
            threshold += step

        return None

    def _optimize_threshold(
        self,
        screener: Screener,
        verifier: Verifier,
        first_threshold: int,
        first_verification: VerificationResult,
        threshold_max: int,
        sample_range: float,
        max_verify_range: float,
        verification_samples: int,
    ) -> int:
        """
        Optimize by searching higher thresholds for better accuracy.

        After finding the first passing threshold, continue searching up to
        20% higher with a z_limit safety bound computed from the first
        verification's probe medians. Returns the threshold with the lowest
        median_range.
        """
        # Already at the natural floor — can't improve
        if first_verification.median_range <= sample_range:
            logger.info(
                "Threshold %d already at optimal accuracy (%smm <= %smm sample range), skipping optimization.",
                first_threshold,
                format_distance(first_verification.median_range),
                format_distance(sample_range),
            )
            return first_threshold

        # Compute z_limit from first verification
        z_limit = min(first_verification.probe_medians) - max_verify_range
        optimization_max = min(int(first_threshold * 1.2), threshold_max)
        screening_samples = self._config.touch.samples + self._config.touch.max_noisy_samples

        logger.info(
            "Optimizing: searching thresholds %d-%d with z_limit=%.4fmm",
            first_threshold,
            optimization_max,
            z_limit,
        )

        best_threshold = first_threshold
        best_range = first_verification.median_range

        threshold = first_threshold + calculate_step(first_threshold, first_verification.median_range, sample_range)
        while threshold <= optimization_max:
            verification, step = self._try_threshold(
                screener,
                verifier,
                threshold,
                screening_samples,
                sample_range,
                max_verify_range,
                verification_samples,
                z_limit=z_limit,
            )

            if verification is None:
                threshold += step
                continue

            if verification.passed(max_verify_range) and verification.median_range < best_range:
                logger.info(
                    "Optimization: threshold %d improved accuracy %smm -> %smm",
                    threshold,
                    format_distance(best_range),
                    format_distance(verification.median_range),
                )
                best_threshold = threshold
                best_range = verification.median_range

                # Definitely satisfied — can't improve beyond per-probe noise
                if best_range <= sample_range:
                    logger.info(
                        "Optimization: reached optimal accuracy (%smm <= %smm sample range), stopping.",
                        format_distance(best_range),
                        format_distance(sample_range),
                    )
                    break

            threshold += calculate_step(threshold, best_range, sample_range)

        if best_threshold != first_threshold:
            logger.info(
                "Optimization complete: threshold %d -> %d (accuracy %smm)",
                first_threshold,
                best_threshold,
                format_distance(best_range),
            )
        else:
            logger.info(
                "Optimization complete: threshold %d remains best (accuracy %smm)",
                best_threshold,
                format_distance(best_range),
            )

        return best_threshold

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

    def perform_touch_probe(self, *, z_limit: float | None = None) -> float:
        """
        Perform one complete touch probe sequence.

        Uses the shared run_probe_sequence to ensure parity with
        the runtime TouchMode._run_probe() logic.
        """
        return run_probe_sequence(
            lambda: self._perform_single_probe(z_limit=z_limit),
            samples=self._config.samples,
            max_samples=self._config.max_samples,
            max_window=self._config.max_window,
            sample_range=self._config.sample_range,
        )
