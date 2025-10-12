from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cartographer.coil.helpers import line0, line120, line_fit, param_linear
from cartographer.interfaces.configuration import CoilCalibrationConfiguration
from cartographer.lib.scipy_helpers import curve_fit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cartographer.interfaces.printer import CoilCalibrationReference, Sample


def fit_coil_temperature_model(
    data_per_height: dict[float, list[Sample]], ref: CoilCalibrationReference
) -> CoilCalibrationConfiguration:
    """
    Fits a coil temperature compensation model across multiple probe heights.

    For each height, this function:
    1. Processes temperature-frequency sample data to extract quadratic coefficients
    2. Fits linear relationships between frequency and the quadratic coefficients
    3. Returns calibration parameters for temperature compensation

    Args:
        data_per_height: Dictionary mapping probe heights (mm) to lists of temperature/frequency samples
        ref: Reference configuration containing baseline frequency values

    Returns:
        CoilCalibrationConfiguration containing the linear relationship parameters
        for temperature compensation coefficients 'a' and 'b'
    """
    coefficients_a: list[float] = []
    coefficients_b: list[float] = []
    frequencies: list[float] = []

    for _, samples in data_per_height.items():
        (a, b, freq_at_vertex) = _process_samples(samples)
        coefficients_a.append(a)  # 'a' coefficient from quadratic fit
        coefficients_b.append(b)  # 'b' coefficient from quadratic fit
        frequencies.append(freq_at_vertex)  # frequency at quadratic vertex

    # Fit linear relationship: coefficient_a = linear_a * (freq - min_freq) + linear_b
    freq_array: NDArray[np.float_] = np.asarray(frequencies) - ref.min_frequency
    linear_params_a, _ = curve_fit(param_linear, freq_array, coefficients_a, maxfev=100000, ftol=1e-10, xtol=1e-10)

    # Fit linear relationship: coefficient_b = linear_a * (freq - min_freq) + linear_b
    linear_params_b, _ = curve_fit(param_linear, freq_array, coefficients_b, maxfev=100000, ftol=1e-10, xtol=1e-10)

    return CoilCalibrationConfiguration(
        a_a=linear_params_a[0],  # Slope for 'a' coefficient vs frequency
        a_b=linear_params_a[1],  # Intercept for 'a' coefficient vs frequency
        b_a=linear_params_b[0],  # Slope for 'b' coefficient vs frequency
        b_b=linear_params_b[1],  # Intercept for 'b' coefficient vs frequency
    )


def _downsample_by_temperature(samples: list[Sample], target_count: int) -> list[Sample]:
    """Downsample samples to ensure even distribution across temperature ranges."""
    if len(samples) <= target_count:
        return samples

    temperatures = [s.temperature for s in samples]
    temp_min, temp_max = min(temperatures), max(temperatures)

    # Create temperature bins
    n_bins = 10  # Adjust based on your needs
    bin_width = (temp_max - temp_min) / n_bins
    samples_per_bin = target_count // n_bins

    binned_samples: list[list[Sample]] = [[] for _ in range(n_bins)]

    # Sort samples into temperature bins
    for sample in samples:
        bin_idx = min(int((sample.temperature - temp_min) / bin_width), n_bins - 1)
        binned_samples[bin_idx].append(sample)

    # Sample evenly from each bin
    downsampled: list[Sample] = []
    for bin_samples in binned_samples:
        if not bin_samples:
            continue
        # Take evenly spaced samples from this bin
        step = max(1, len(bin_samples) // samples_per_bin)
        downsampled.extend(bin_samples[::step][:samples_per_bin])

    return downsampled


def _process_samples(samples: list[Sample]) -> tuple[float, float, float]:
    """
    Processes temperature-frequency samples to extract quadratic relationship coefficients.

    Fits a quadratic function: frequency = a*temperature² + b*temperature + c

    The function handles three cases based on where the quadratic's vertex falls:
    - Normal case (vertex 0-120°C): Uses full quadratic fit
    - Hot case (vertex >120°C): Constrains vertex to 120°C using line120
    - Cold case (vertex <0°C): Constrains vertex to 0°C using line0

    Args:
        samples: List of Sample objects containing temperature and frequency data

    Returns:
        Tuple containing (a_coefficient, b_coefficient, frequency_at_vertex)

    Raises:
        RuntimeError: If fewer than 300 samples provided (insufficient for calibration)
    """
    if len(samples) < 300:
        msg = f"Insufficient samples for calibration: {len(samples)} (need at least 300)"
        raise RuntimeError(msg)

    # Downsample if we have too many samples to improve processing speed
    if len(samples) > 1000:
        samples = _downsample_by_temperature(samples, target_count=800)

    frequencies: list[float] = [s.frequency for s in samples]
    temperatures: list[float] = [s.temperature for s in samples]

    # Fit quadratic: freq = a*temp² + b*temp + c
    param_bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    [a, b, c], _ = curve_fit(
        line_fit, temperatures, frequencies, bounds=param_bounds, maxfev=100000, ftol=1e-10, xtol=1e-10
    )

    # Calculate vertex position of the quadratic: x = -b/(2a)
    vertex_temperature = -b / (2 * a)

    # Handle constrained cases based on vertex position
    if vertex_temperature > 120:
        # Vertex too hot - constrain to 120°C
        [constrained_a, constrained_b], _ = curve_fit(
            line120,
            temperatures,
            frequencies,
            bounds=([0, -np.inf], [np.inf, np.inf]),
            maxfev=100000,
            ftol=1e-10,
            xtol=1e-10,
        )
        return (
            constrained_a,  # a coefficient
            -240 * constrained_a,  # b coefficient (from line120 constraint)
            line120(120, constrained_a, constrained_b),  # freq at vertex (120°C)
        )

    elif vertex_temperature < 0:
        # Vertex too cold - constrain to 0°C
        [constrained_a, constrained_b], _ = curve_fit(
            line0,
            temperatures,
            frequencies,
            bounds=([0, -np.inf], [np.inf, np.inf]),
            maxfev=100000,
            ftol=1e-10,
            xtol=1e-10,
        )
        return (
            constrained_a,  # a coefficient
            0,  # b coefficient (from line0 constraint)
            line0(0, constrained_a, constrained_b),  # freq at vertex (0°C)
        )

    # Normal case - vertex within reasonable range (0-120°C)
    # Calculate frequency at the vertex and return coefficients
    frequency_at_vertex = line_fit(vertex_temperature, a, b, c)
    return (a, b, frequency_at_vertex)
