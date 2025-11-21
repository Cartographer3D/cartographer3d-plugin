from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.configuration import (
    Configuration,
    TouchModelConfiguration,
)
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.lib.statistics import compute_mad
from cartographer.macros.utils import force_home_z
from cartographer.probe.touch_mode import (
    MAD_TOLERANCE,
    TouchMode,
    TouchModeConfiguration,
)

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


logger = logging.getLogger(__name__)


MIN_ALLOWED_STEP = 75
MAX_ALLOWED_STEP = 500
DEFAULT_TOUCH_MODEL_NAME = "default"
DEFAULT_Z_OFFSET = -0.05

# Stop after this many consecutive unacceptable results past the
# acceptable range
MAX_CONSECUTIVE_UNACCEPTABLE = 2
# Don't continue more than this percentage past first acceptable
MAX_SEARCH_RANGE_MULTIPLIER = 1.20


@dataclass(frozen=True)
class CalibrationResult:
    """Result from testing a single threshold value."""

    threshold: int
    score: float
    samples: tuple[float, ...]

    @property
    def is_acceptable(self) -> bool:
        """Check if score is within tolerance."""
        return self.score <= MAD_TOLERANCE


@dataclass
class CalibrationState:
    """Tracks state during threshold calibration."""

    first_acceptable: int | None = None
    last_acceptable: int | None = None
    consecutive_unacceptable: int = 0
    threshold_max: int = 0


def compute_step_increase(current_threshold: int, score: float) -> int:
    """
    Calculate the next step size based on current threshold and score.

    Uses an adaptive algorithm that takes larger steps when far from
    the target and smaller steps when close.
    """
    allowed_step = (current_threshold * 0.06) ** 1.15
    allowed_max_step = max(MIN_ALLOWED_STEP, min(MAX_ALLOWED_STEP, allowed_step))

    ratio = max(1.0, score / MAD_TOLERANCE)
    raw_step = allowed_max_step * (1 - (1 / ratio))

    return int(round(min(allowed_max_step, max(MIN_ALLOWED_STEP, raw_step))))


def format_score_message(score: float) -> str:
    """
    Format score as user-friendly message with visual indicator.

    Parameters
    ----------
    score : float
        The MAD score to format

    Returns
    -------
    str
        Formatted message with quality indicator
    """
    ratio = score / MAD_TOLERANCE

    if ratio <= 1.0:
        percent_of_target = int(ratio * 100)
        indicator = "✓"
        quality = "good"
        if ratio <= 0.5:
            indicator = "✓✓"
            quality = "excellent"
        if ratio <= 0.1:
            indicator = "✓✓✓"
            quality = "perfect"
        return f"{indicator} {quality} (score={score:.6f}, {percent_of_target}% of limit)"

    # Above tolerance - determine quality level
    if ratio <= 1.5:
        indicator = "✗"
        quality = "marginal"
    elif ratio <= 3.0:
        indicator = "✗✗"
        quality = "poor"
    else:
        indicator = "✗✗✗"
        quality = "very poor"

    times_over = ratio
    return f"{indicator} {quality} (score={score:.6f}, {times_over:.1f}x over limit)"


@final
class TouchCalibrateMacro(Macro):
    description = "Run the touch calibration"

    def __init__(
        self,
        probe: Probe,
        mcu: Mcu,
        toolhead: Toolhead,
        config: Configuration,
    ) -> None:
        self._probe = probe
        self._mcu = mcu
        self._toolhead = toolhead
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        name = params.get("MODEL", DEFAULT_TOUCH_MODEL_NAME).lower()
        speed = params.get_int("SPEED", default=2, minval=1, maxval=5)
        threshold_start = params.get_int("START", default=500, minval=100)
        threshold_max = params.get_int("MAX", default=3000, minval=threshold_start)

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

        logger.info(
            "Starting touch calibration at speed %d, threshold %d-%d",
            speed,
            threshold_start,
            threshold_max,
        )
        logger.info("Target score: ≤%.6f (lower is better)", MAD_TOLERANCE)

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=threshold_start,
            speed=speed,
        )

        with force_home_z(self._toolhead):
            threshold = self._find_optimal_threshold(
                calibration_mode,
                threshold_start,
                threshold_max,
            )

        if threshold is None:
            logger.info(
                "Failed to calibrate (thresholds %d-%d).\n"
                "Try increasing MAX.\n"
                "CARTOGRAPHER_TOUCH_CALIBRATE START=%d MAX=%d",
                threshold_start,
                threshold_max,
                threshold_max,
                threshold_max + 2000,
            )
            return

        logger.info(
            "Successfully calibrated (threshold %d, speed %.1f)",
            threshold,
            speed,
        )
        model = TouchModelConfiguration(name, threshold, speed, DEFAULT_Z_OFFSET)
        self._config.save_touch_model(model)
        self._probe.touch.load_model(name)
        logger.info(
            "Touch model %s has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config "
            "file and restart the printer.",
            name,
        )

    def _find_optimal_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold_start: int,
        threshold_max: int,
    ) -> int | None:
        """
        Find optimal threshold by testing until we find a stable range.

        Strategy:
        1. Test thresholds with adaptive steps until acceptable
        2. Continue testing to find the full acceptable range
        3. Stop when we've gone too far or hit limits
        4. Choose the midpoint of the acceptable range
        """
        state = CalibrationState(threshold_max=threshold_max)
        current_threshold = threshold_start

        while current_threshold <= state.threshold_max:
            result = self._test_threshold(calibration_mode, current_threshold)
            logger.info(
                "Threshold %d: %s",
                result.threshold,
                format_score_message(result.score),
            )

            self._update_state(state, result)

            if self._should_stop_calibration(state):
                break

            current_threshold = self._get_next_threshold(
                current_threshold,
                result.score,
                state,
            )

        return self._calculate_optimal(state)

    def _update_state(
        self,
        state: CalibrationState,
        result: CalibrationResult,
    ) -> None:
        """Update calibration state based on current result."""
        if not result.is_acceptable:
            if state.first_acceptable is not None:
                state.consecutive_unacceptable += 1
            return

        # Result is acceptable
        state.last_acceptable = result.threshold
        state.consecutive_unacceptable = 0
        if state.first_acceptable is not None:
            return

        state.first_acceptable = result.threshold
        # Limit search to 20% past first acceptable
        state.threshold_max = min(
            int(state.first_acceptable * MAX_SEARCH_RANGE_MULTIPLIER),
            state.threshold_max,
        )
        logger.debug(
            "Found first acceptable threshold: %d (will search up to %d)",
            state.first_acceptable,
            state.threshold_max,
        )

    def _should_stop_calibration(self, state: CalibrationState) -> bool:
        """Determine if calibration should stop early."""
        if state.first_acceptable is None:
            return False
        return state.consecutive_unacceptable >= MAX_CONSECUTIVE_UNACCEPTABLE

    def _get_next_threshold(
        self,
        current_threshold: int,
        score: float,
        state: CalibrationState,
    ) -> int:
        """Calculate the next threshold to test."""
        if state.first_acceptable is None:
            # Not yet acceptable - use adaptive large steps
            step = compute_step_increase(current_threshold, score)
        else:
            # Found acceptable - use small percentage-based steps
            step = max(MIN_ALLOWED_STEP, int(current_threshold * 0.05))

        next_threshold = min(current_threshold + step, state.threshold_max)
        actual_step = next_threshold - current_threshold

        if actual_step < MIN_ALLOWED_STEP:
            logger.debug(
                "Stopping: remaining step size (%d) below minimum (%d)",
                actual_step,
                MIN_ALLOWED_STEP,
            )
            # Force loop exit by returning value past max
            return state.threshold_max + 1

        logger.debug("Next threshold: %d (+%d)", next_threshold, actual_step)
        return next_threshold

    def _calculate_optimal(self, state: CalibrationState) -> int | None:
        """Calculate optimal threshold from calibration state."""
        if state.first_acceptable is None:
            return None

        # Found complete range
        if state.last_acceptable is not None and state.last_acceptable > state.first_acceptable:
            optimal = (state.first_acceptable + state.last_acceptable) // 2
            logger.info(
                "Found acceptable range: %d-%d, using midpoint: %d",
                state.first_acceptable,
                state.last_acceptable,
                optimal,
            )
            return optimal

        # Only found first acceptable - use midpoint to limit
        optimal = (state.first_acceptable + state.threshold_max) // 2
        logger.warning(
            "Hit search limit at threshold %d. Using midpoint between first acceptable (%d) and search limit: %d",
            state.threshold_max,
            state.first_acceptable,
            optimal,
        )
        logger.info(
            "Consider refining calibration with:\nCARTOGRAPHER_TOUCH_CALIBRATE START=%d MAX=%d",
            state.first_acceptable,
            min(
                int(state.first_acceptable * 1.5),
                state.threshold_max + 1000,
            ),
        )
        return optimal

    def _test_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold: int,
    ) -> CalibrationResult:
        """
        Test a single threshold value and return the result.

        Collects samples and computes MAD score to determine
        acceptability.
        """
        samples = calibration_mode.collect_samples(threshold)
        score = compute_mad(samples)

        logger.debug(
            "Threshold %d: samples=%s, score=%.6f",
            threshold,
            ", ".join(f"{s:.6f}" for s in samples[:5]) + (", ..." if len(samples) > 5 else ""),
            score,
        )

        return CalibrationResult(
            threshold=threshold,
            score=score,
            samples=samples,
        )


@final
class CalibrationTouchMode(TouchMode):
    """Touch mode configured specifically for calibration."""

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
        """Update the threshold for the calibration model."""
        self._models["calibration"] = replace(
            self._models["calibration"],
            threshold=threshold,
        )
        self.load_model("calibration")

    def collect_samples(self, threshold: int) -> tuple[float, ...]:
        """
        Collect samples at the given threshold.

        Performs multiple touches and returns sorted results.
        Includes early stopping if results are clearly too poor.
        """
        samples: list[float] = []
        self.set_threshold(threshold)

        max_samples = self._config.samples * 3

        for _ in range(max_samples):
            pos = self._perform_single_probe()
            samples.append(pos)

            if self._should_stop_early(samples):
                break

        return tuple(sorted(samples))

    def _should_stop_early(self, samples: list[float]) -> bool:
        """
        Determine if we should stop collecting samples early.

        Returns True if we have enough samples and the score is
        significantly worse than tolerance (5x).
        """
        if len(samples) < self._config.samples:
            return False
        score = compute_mad(samples)
        if score > MAD_TOLERANCE * 5:
            logger.debug(
                "Early stop at sample %d (score %.6f >> %.6f)",
                len(samples),
                score,
                MAD_TOLERANCE,
            )
            return True
        return False
