from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import numpy as np

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import MacroParams, Mcu, Sample, Toolhead
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)

# The threshold noise safety margin
SAFETY_MARGIN_PERCENTAGE = 0.05


@dataclass
class FrequencyAnalysisResult:
    """Results from frequency-based touch detection analysis."""

    recommended_threshold: int
    max_noise_variation: float
    signal_strength: float
    confidence: float


def analyze_touch_characteristics(samples: list[Sample]) -> FrequencyAnalysisResult:
    """
    Analyze count changes during descent to determine optimal touch threshold.

    The MCU firmware:
    1. Calculates deltas between consecutive samples
    2. Maintains a rolling average of the last 6 deltas
    3. Tracks the maximum average seen
    4. Triggers when: max_average > threshold + current_average

    The threshold should be set just above the maximum noise variation.

    Parameters
    ----------
    samples : list[Sample]
        Samples collected during controlled descent from z=5 to z=-0.5

    Returns
    -------
    FrequencyAnalysisResult
        Recommended threshold and diagnostic information

    Raises
    ------
    RuntimeError
        If insufficient samples with valid positions
    """
    if len(samples) < 500:
        msg = "Insufficient samples for frequency analysis"
        raise RuntimeError(msg)

    # Extract z positions and raw counts
    counts_list: list[int] = [sample.count for sample in samples]
    counts = np.array(counts_list)

    # Calculate deltas (what MCU puts in stack)
    deltas = np.diff(counts)

    # Calculate rolling averages with window size 6 (what MCU calls 'avr')
    window_size = 6
    rolling_avgs = np.convolve(deltas, np.ones(window_size) / window_size, mode="valid")

    # Analyze the rolling averages to find noise floor
    # Use the first 50% of samples as the "free space" region
    # This assumes descent starts in free space
    free_space_count = int(len(rolling_avgs) * 0.5)
    free_space_avgs = rolling_avgs[:free_space_count]

    # Calculate noise characteristics from free-space region
    max_noise_variation = float(np.ptp(free_space_avgs))  # Peak-to-peak
    noise_std = float(np.std(free_space_avgs))

    # Recommended threshold calculation:
    recommended_threshold = int(round(max_noise_variation * (1 + SAFETY_MARGIN_PERCENTAGE)))

    # Ensure threshold is at least 3x the noise std dev for statistical confidence
    min_threshold = int(round(noise_std * 3))
    recommended_threshold = max(recommended_threshold, min_threshold)

    # For confidence calculation, look at the entire signal
    # The signal strength is the maximum variation observed
    signal_strength = float(np.ptp(rolling_avgs))

    # Calculate confidence based on signal-to-noise ratio
    snr = signal_strength / (max_noise_variation + 1e-6)
    confidence = min(1.0, snr / 3.0)  # SNR of 3+ = 100% confidence

    return FrequencyAnalysisResult(
        recommended_threshold=recommended_threshold,
        max_noise_variation=max_noise_variation,
        signal_strength=signal_strength,
        confidence=confidence,
    )


@final
class FrequencyAnalysisTouchCalibrateMethod:
    """
    Calibrate touch threshold using frequency response analysis.

    This method performs a controlled descent while monitoring count changes.
    """

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

    def run(self, params: MacroParams) -> tuple[int, float] | None:
        """
        Execute frequency analysis calibration.

        Parameters
        ----------
        params : MacroParams
            Macro parameters. Supported options:
            - MODEL: Name for the touch model (default: "default")
            - SPEED: Descent speed in mm/s (default: 2.0)
            - START_Z: Starting height in mm (default: 5.0)
            - END_Z: Ending height in mm (default: -0.5)
        """
        speed = params.get_float("SPEED", default=2.0, minval=1.0, maxval=5.0)
        start_z = 5.0
        end_z = params.get_float("END_Z", default=-0.5, minval=-1.0, maxval=0.0)

        # Ensure we're homed
        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y") or not self._toolhead.is_homed("z"):
            msg = "Must home before calibration"
            raise RuntimeError(msg)

        # Move to calibration position
        self._toolhead.move(z=start_z, speed=5.0)
        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()
        self._toolhead.dwell(1.0)

        logger.info(
            "Starting frequency analysis descent from z=%.1f to z=%.1f at %.1f mm/s...",
            start_z,
            end_z,
            speed,
        )

        with self._probe.scan.start_session() as session:
            # Wait for initial samples to stabilize
            session.wait_for(lambda samples: len(samples) >= 10)

            # Perform controlled descent
            self._toolhead.move(z=end_z, speed=speed)
            self._toolhead.wait_moves()

            # Collect additional samples after move completes
            time = self._toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 10)

        samples = session.get_items()
        logger.info("Collected %d samples during descent", len(samples))
        # Move back to safe height
        self._toolhead.move(z=start_z, speed=5.0)

        result = analyze_touch_characteristics(samples)

        logger.info(
            "Frequency analysis results:\n"
            "Max noise variation: %.2f\n"
            "Signal strength: %.2f\n"
            "Signal-to-noise ratio: %.1f:1\n"
            "Recommended threshold: %d\n"
            "Confidence: %.1f%%",
            result.max_noise_variation,
            result.signal_strength,
            result.signal_strength / result.max_noise_variation if result.max_noise_variation > 0 else 0,
            result.recommended_threshold,
            result.confidence * 100,
        )

        # Warn if confidence is low
        if result.confidence < 0.5:
            logger.warning(
                "Moderate confidence (%.1f%%) in calibration.\n"
                "Signal-to-noise ratio is %.1f:1.\n"
                "Consider running calibration again to verify consistency.",
                result.confidence * 100,
                result.signal_strength / result.max_noise_variation if result.max_noise_variation > 0 else 0,
            )

        return result.recommended_threshold, speed
