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

# Analyze noise from the middle portion of descent, close to where touch occurs
# This gives us noise characteristics in the actual operating range
NOISE_ANALYSIS_START_PERCENTILE = 0.20  # Skip initial 20% (far from bed)
NOISE_ANALYSIS_END_PERCENTILE = 0.60  # Use up to 60% of descent

# Safety margin: how much above the maximum observed noise to set threshold
SAFETY_MARGIN_PERCENTAGE = 0.10  # 10% above max noise


@dataclass
class FrequencyAnalysisResult:
    """Results from frequency-based touch detection analysis."""

    recommended_threshold: int
    max_noise_variation: float
    noise_std: float


def analyze_touch_characteristics(samples: list[Sample]) -> FrequencyAnalysisResult:
    """
    Analyze count changes during descent to determine optimal touch threshold.

    The MCU firmware:
    1. Calculates deltas between consecutive samples: `data - homing_freq`
    2. Maintains a rolling average of the last 6 deltas in a stack
    3. Tracks the maximum average seen so far: `max`
    4. Triggers when: `max > trigger_threshold + current_average && max > 100`

    The threshold represents the minimum change in the rolling average (above the
    current average) needed to trigger touch detection.

    We analyze noise from the middle portion of the descent (20%-60%) to get
    noise characteristics that are representative of the operating range where
    touch detection actually occurs.

    Parameters
    ----------
    samples : list[Sample]
        Samples collected during controlled descent

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

    # Extract raw counts
    counts_list: list[int] = [sample.count for sample in samples]
    counts = np.array(counts_list)

    # Calculate deltas (what MCU puts in stack array)
    # These are the consecutive count changes
    deltas = np.diff(counts)

    # Calculate rolling averages with window size 6 (what MCU calls 'avr')
    # This simulates the exact filtering the MCU firmware applies
    window_size = 6
    rolling_avgs = np.convolve(deltas, np.ones(window_size) / window_size, mode="valid")

    # Identify noise analysis region from the middle portion of descent
    # This is close enough to the bed to be representative, but not yet touching
    start_idx = int(len(rolling_avgs) * NOISE_ANALYSIS_START_PERCENTILE)
    end_idx = int(len(rolling_avgs) * NOISE_ANALYSIS_END_PERCENTILE)
    noise_region_avgs = rolling_avgs[start_idx:end_idx]

    if len(noise_region_avgs) < 50:
        msg = "Insufficient samples in noise analysis region"
        raise RuntimeError(msg)

    # Calculate noise characteristics from this region
    # Peak-to-peak gives us the maximum variation we see in normal operation
    max_noise_variation = float(np.ptp(noise_region_avgs))
    noise_std = float(np.std(noise_region_avgs))

    # Recommended threshold calculation:
    # Set threshold just above the maximum noise variation with a safety margin
    recommended_threshold_float = max_noise_variation * (1 + SAFETY_MARGIN_PERCENTAGE)

    # Also ensure threshold is at least 3x the standard deviation
    # This provides statistical confidence that we're above random fluctuations
    min_threshold = int(round(noise_std * 3))
    recommended_threshold = max(int(round(recommended_threshold_float)), min_threshold)

    return FrequencyAnalysisResult(
        recommended_threshold=recommended_threshold,
        max_noise_variation=max_noise_variation,
        noise_std=noise_std,
    )


@final
class FrequencyAnalysisTouchCalibrateMethod:
    """
    Calibrate touch threshold using frequency response analysis.
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
            - SPEED: Descent speed in mm/s (default: 2.0)
            - END_Z: Ending height in mm (default: 1.0)
        """
        speed = params.get_float("SPEED", default=2.0, minval=1.0, maxval=5.0)
        start_z = 5.0
        end_z = 0.5

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
        logger.debug("Collected %d samples during descent", len(samples))

        # Move back to safe height before analysis
        self._toolhead.move(z=start_z, speed=5.0)

        result = analyze_touch_characteristics(samples)

        logger.info(
            "Frequency analysis results:\n"
            "Max noise variation: %.2f\n"
            "Noise std deviation: %.2f\n"
            "Recommended threshold: %d",
            result.max_noise_variation,
            result.noise_std,
            result.recommended_threshold,
        )

        return result.recommended_threshold, speed
