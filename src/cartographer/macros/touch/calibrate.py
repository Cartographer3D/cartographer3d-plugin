from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.configuration import (
    Configuration,
    TouchModelConfiguration,
)
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.utils import force_home_z
from cartographer.probe.touch_mode import (
    MAX_SAMPLE_RANGE,
    TouchMode,
    TouchModeConfiguration,
    compute_range,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


logger = logging.getLogger(__name__)


MIN_STEP = 50
DEFAULT_TOUCH_MODEL_NAME = "default"
DEFAULT_Z_OFFSET = -0.05

# How many top subsets to consider for median calculation
TOP_SUBSET_COUNT = 10


@dataclass(frozen=True)
class ThresholdResult:
    """Result from testing a single threshold."""

    threshold: int
    samples: tuple[float, ...]
    best_subset: tuple[float, ...] | None
    best_range: float
    median_range: float

    @property
    def is_consistent(self) -> bool:
        """
        Check if the threshold produces consistent results.

        Uses the median range of the top subsets rather than just
        the best, to ensure we can reliably find a good subset.
        """
        return self.median_range <= MAX_SAMPLE_RANGE


def analyze_subsets(
    samples: Sequence[float],
    subset_size: int,
    top_n: int = TOP_SUBSET_COUNT,
) -> tuple[tuple[float, ...] | None, float, float]:
    """
    Analyze subsets to find the best and compute statistics.

    Returns (best_subset, best_range, median_range, total_count).
    """
    top_subsets = heapq.nsmallest(
        top_n,
        combinations(samples, subset_size),  # Generator, not list
        key=compute_range,
    )

    if not top_subsets:
        return None, float("inf"), float("inf")

    best = top_subsets[0]
    best_range = compute_range(best)

    # Compute median range of top N subsets
    top_count = min(top_n, len(top_subsets))
    top_ranges = [compute_range(s) for s in top_subsets[:top_count]]
    median_range = float(np.median(top_ranges))

    return best, best_range, median_range


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
        speed = params.get_int("SPEED", default=3, minval=1, maxval=5)
        threshold_start = params.get_int("START", default=500, minval=100)
        threshold_max = params.get_int("MAX", default=5000, minval=threshold_start)

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

        required_samples = self._config.touch.samples
        max_samples = self._config.touch.max_samples

        logger.info(
            "Starting touch calibration (speed=%d, range=%d-%d)",
            speed,
            threshold_start,
            threshold_max,
        )
        logger.info(
            "Looking for %d samples within %.3fmm range (max %d attempts)",
            required_samples,
            MAX_SAMPLE_RANGE,
            max_samples,
        )

        calibration_mode = CalibrationTouchMode(
            self._mcu,
            self._toolhead,
            TouchModeConfiguration.from_config(self._config),
            threshold=threshold_start,
            speed=speed,
        )

        with force_home_z(self._toolhead):
            threshold = self._find_minimum_threshold(
                calibration_mode,
                threshold_start,
                threshold_max,
                required_samples,
                max_samples,
            )

        if threshold is None:
            logger.info(
                "Failed to find reliable threshold in range %d-%d.\n"
                "Try increasing MAX:\n"
                "CARTOGRAPHER_TOUCH_CALIBRATE START=%d MAX=%d",
                threshold_start,
                threshold_max,
                threshold_max,
                threshold_max + 2000,
            )
            return

        logger.info(
            "Calibration complete: threshold=%d, speed=%d",
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

    def _find_minimum_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold_start: int,
        threshold_max: int,
        required_samples: int,
        max_samples: int,
    ) -> int | None:
        """
        Find the minimum threshold that produces consistent results.

        Strategy:
        1. Linear search to find first consistent threshold
        2. Verify with more samples
        3. If verification fails, step up and retry
        """
        threshold = threshold_start

        while threshold <= threshold_max:
            # Quick test with required sample count
            result = self._test_threshold(
                calibration_mode,
                threshold,
                required_samples,
            )

            self._log_result(result)
            self._log_result_debug(result)

            if not result.is_consistent:
                threshold += self._calculate_step(threshold, result.median_range)
                continue

            # Verify with more samples
            logger.info(
                "Threshold %d looks promising, verifying...",
                threshold,
            )
            verification = self._test_threshold(
                calibration_mode,
                threshold,
                max_samples,
            )

            self._log_result(verification, prefix="Verification")
            self._log_result_debug(verification, prefix="Verification")

            if verification.is_consistent:
                logger.info(
                    "Threshold %d verified: best=%.4fmm, median=%.4fmm",
                    threshold,
                    verification.best_range,
                    verification.median_range,
                )
                return threshold

            # Verification failed - step up
            logger.debug(
                "Verification failed (median=%.4f > %.4f), stepping up",
                verification.median_range,
                MAX_SAMPLE_RANGE,
            )
            threshold += MIN_STEP

        return None

    def _test_threshold(
        self,
        calibration_mode: CalibrationTouchMode,
        threshold: int,
        sample_count: int,
    ) -> ThresholdResult:
        """Test a threshold by collecting samples and analyzing subsets."""
        samples = calibration_mode.collect_samples(threshold, sample_count)
        required = self._config.touch.samples

        # Analyze subsets
        best, best_range, median_range = analyze_subsets(samples, required)

        return ThresholdResult(
            threshold=threshold,
            samples=samples,
            best_subset=best,
            best_range=best_range,
            median_range=median_range,
        )

    def _calculate_step(self, threshold: int, range_value: float) -> int:
        """
        Calculate step size based on how far from target we are.

        Larger steps when range is very bad, smaller steps when close.
        """
        if range_value > MAX_SAMPLE_RANGE * 10:
            return max(MIN_STEP, int(threshold * 0.10))
        if range_value > MAX_SAMPLE_RANGE * 3:
            return max(MIN_STEP, int(threshold * 0.05))
        return MIN_STEP

    def _log_result(
        self,
        result: ThresholdResult,
        prefix: str = "Threshold",
    ) -> None:
        """Log a threshold test result at INFO level."""
        status = "✓" if result.is_consistent else "✗"
        logger.info(
            "%s %d: %s best=%.4fmm, median=%.4fmm (%d samples)",
            prefix,
            result.threshold,
            status,
            result.best_range,
            result.median_range,
            len(result.samples),
        )

    def _log_result_debug(
        self,
        result: ThresholdResult,
        prefix: str = "Threshold",
    ) -> None:
        """Log detailed threshold test info at DEBUG level."""
        samples_str = ", ".join(f"{s:.4f}" for s in result.samples)
        best_str = ", ".join(f"{s:.4f}" for s in result.best_subset) if result.best_subset else "none"

        logger.debug(
            "%s %d details:\n"
            "  samples: [%s]\n"
            "  best subset: [%s]\n"
            "  best range: %.4f mm\n"
            "  median range (top %d): %.4f mm\n",
            prefix,
            result.threshold,
            samples_str,
            best_str,
            result.best_range,
            TOP_SUBSET_COUNT,
            result.median_range,
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
