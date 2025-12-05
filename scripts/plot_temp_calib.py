# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import csv
import os
import re
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from cartographer.coil.calibration import fit_coil_temperature_model
from cartographer.coil.temperature_compensation import CoilReferenceMcu, CoilTemperatureCompensationModel
from cartographer.interfaces.printer import CoilCalibrationReference, Position, Sample

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class StubMcu(CoilReferenceMcu):
    @override
    def get_coil_reference(self) -> CoilCalibrationReference:
        return CoilCalibrationReference(min_frequency=2943054, min_frequency_temperature=23)


mcu = StubMcu()


def read_samples_from_csv(file_path: str) -> list[Sample]:
    samples: list[Sample] = []

    with open(file_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            x = float(row["position_x"])
            y = float(row["position_y"])
            z = float(row["position_z"])
            sample = Sample(
                raw_count=0,
                time=float(row["time"]),
                frequency=float(row["frequency"]),
                temperature=float(row["temperature"]),
                position=Position(x, y, z),
            )
            samples.append(sample)

    return samples


def load_data_from_directory(directory: str) -> dict[float, list[Sample]]:
    data_per_height: dict[float, list[Sample]] = {}

    if not os.path.isdir(directory):
        msg = f"Directory does not exist: {directory}"
        raise ValueError(msg)

    # Pattern to match files with height information
    # Matches patterns like: cartographer_temp_calib_h1mm_20250929_085410.csv
    height_pattern = re.compile(r"cartographer_temp_calib_h(\d)mm", re.IGNORECASE)

    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    if not csv_files:
        msg = f"No CSV files found in directory: {directory}"
        raise ValueError(msg)

    found_files = 0
    for filename in csv_files:
        match = height_pattern.search(filename)
        if match:
            height = float(match.group(1))
            file_path = os.path.join(directory, filename)

            try:
                samples = read_samples_from_csv(file_path)
                data_per_height[height] = samples
                print(f"Loaded {len(samples)} samples from {filename} (height: {height}mm)")
                found_files += 1
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    if found_files == 0:
        print("Available CSV files:")
        for f in csv_files:
            print(f"  - {f}")
        msg = "No CSV files with recognizable height pattern found. Expected pattern: 'cartographer_temp_calib_h{d}mm'"
        raise ValueError(msg)

    print(f"Successfully loaded data for {len(data_per_height)} heights")
    return data_per_height


def normalize_frequencies(frequencies: list[float]) -> list[float]:
    """Normalize frequencies by removing the mean (height-dependent baseline)"""
    mean_freq = np.mean(frequencies)
    return [f - float(mean_freq) for f in frequencies]


def plot_all_samples(
    ax: Axes, data_per_height: dict[float, list[Sample]], model: CoilTemperatureCompensationModel
) -> None:
    for samples in data_per_height.values():
        temperatures = [sample.temperature for sample in samples]
        frequencies = normalize_frequencies([sample.frequency for sample in samples])
        ax.scatter(temperatures, frequencies, alpha=0.6, color="tab:blue")
    for samples in data_per_height.values():
        temperatures = [sample.temperature for sample in samples]
        mean_freq = np.mean([sample.frequency for sample in samples])
        compensated_frequencies = [
            model.compensate(sample.frequency, sample.temperature, 50) - mean_freq for sample in samples
        ]
        ax.scatter(temperatures, compensated_frequencies, alpha=0.6, color="tab:orange")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency vs Temperature")
    ax.grid(True, alpha=0.3)


def plot_compensation_magnitude(
    ax: Axes, data_per_height: dict[float, list[Sample]], model: CoilTemperatureCompensationModel
) -> None:
    """Plot the magnitude of compensation applied at different temperatures"""
    reference_temp = 50.0

    colors = plt.colormaps["viridis"](np.linspace(0, 1, len(data_per_height)))
    for i, (height, samples) in enumerate(sorted(data_per_height.items())):
        temperatures = [sample.temperature for sample in samples]
        compensation_deltas = [
            sample.frequency - model.compensate(sample.frequency, sample.temperature, reference_temp)
            for sample in samples
        ]

        # Sort by temperature for smooth line
        temp_comp_pairs = sorted(zip(temperatures, compensation_deltas))
        sorted_temps, sorted_comps = zip(*temp_comp_pairs)

        ax.plot(sorted_temps, sorted_comps, "o-", label=f"{height}mm", alpha=0.7, color=colors[i])

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Compensation Applied (Hz)")
    ax.set_title("Temperature Compensation Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)


def plot_samples(ax: Axes, samples: list[Sample], label: str, model: CoilTemperatureCompensationModel) -> None:
    temperatures = [sample.temperature for sample in samples]
    frequencies = [sample.frequency for sample in samples]
    compensated_frequencies = [model.compensate(sample.frequency, sample.temperature, 50) for sample in samples]

    ax.scatter(temperatures, frequencies, alpha=0.6, label=label)
    ax.scatter(temperatures, compensated_frequencies, alpha=0.6, label=label + " (compensated)")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency vs Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()


def analyze_model(model: CoilTemperatureCompensationModel, data_per_height: dict[float, list[Sample]]) -> None:
    """
    Analyze how well the temperature compensation model performs.

    This function evaluates the statistical effectiveness of temperature compensation
    by analyzing each height separately, then providing overall metrics about the
    compensation quality across all heights.
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE COMPENSATION MODEL ANALYSIS")
    print("=" * 60)

    reference_temp = 50.0  # Reference temperature for compensation

    # Analyze each height separately and collect improvement metrics
    improvements: list[float] = []
    correlations_before: list[float] = []
    correlations_after: list[float] = []
    correlation_reductions: list[float] = []
    total_samples = 0

    for _, samples in data_per_height.items():
        if len(samples) < 10:
            continue

        # Extract data for this height
        temperatures = np.asarray([sample.temperature for sample in samples])
        raw_frequencies = np.asarray([sample.frequency for sample in samples])
        compensated_frequencies = np.asarray(
            [model.compensate(sample.frequency, sample.temperature, reference_temp) for sample in samples]
        )

        # Calculate statistics for this height
        raw_std = np.std(raw_frequencies)
        comp_std = np.std(compensated_frequencies)
        std_improvement = float((raw_std - comp_std) / raw_std * 100)

        # Temperature correlation analysis
        raw_temp_corr = np.corrcoef(raw_frequencies, temperatures)[0, 1]
        comp_temp_corr = np.corrcoef(compensated_frequencies, temperatures)[0, 1]
        correlation_reduction = abs(raw_temp_corr) - abs(comp_temp_corr)

        improvements.append(std_improvement)
        correlations_before.append(abs(raw_temp_corr))
        correlations_after.append(abs(comp_temp_corr))
        correlation_reductions.append(correlation_reduction)
        total_samples += len(samples)

    # Overall metrics
    avg_improvement = np.mean(improvements)
    avg_corr_reduction = np.mean(correlation_reductions)

    # Overall assessment
    if avg_improvement > 70:
        quality = "Excellent"
    elif avg_improvement > 50:
        quality = "Very Good"
    elif avg_improvement > 30:
        quality = "Good"
    elif avg_improvement > 10:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"  Total samples analyzed: {total_samples}")
    print(f"  Compensation Quality: {quality}")
    print(f"  The model reduces frequency variation by {avg_improvement:.1f}% on average")

    # Temperature dependency assessment based on correlation reduction
    if avg_corr_reduction > 0.3:
        print(f"  ✓ Excellent temperature dependency removal ({avg_corr_reduction:.3f} correlation reduction)")
    elif avg_corr_reduction > 0.1:
        print(f"  ✓ Good temperature dependency removal ({avg_corr_reduction:.3f} correlation reduction)")
    elif avg_corr_reduction > 0.05:
        print(f"  ⚠ Moderate temperature dependency removal ({avg_corr_reduction:.3f} correlation reduction)")
    elif avg_corr_reduction > 0:
        print(f"  ⚠ Limited temperature dependency removal ({avg_corr_reduction:.3f} correlation reduction)")
    else:
        print(f"  ✗ No improvement in temperature dependency ({avg_corr_reduction:.3f} correlation reduction)")

    # Consistency assessment
    if np.std(improvements) < 10:
        print("  ✓ Consistent performance across heights")
    else:
        print("  ⚠ Variable performance across heights")

    # Check if any heights actually got worse
    poor_reductions = [h for h, r in zip(data_per_height.keys(), correlation_reductions) if r < 0]
    if poor_reductions:
        print(f"  ⚠ Heights with increased temperature dependency: {poor_reductions}")


if __name__ == "__main__":
    # Check if directory argument is provided
    if len(sys.argv) != 2:
        print("Usage: python plot_temp_calib.py <directory_path>")
        print("Example: python plot_temp_calib.py ./scripts/")
        sys.exit(1)

    directory = sys.argv[1]
    try:
        # Load data from directory automatically
        data_per_height = load_data_from_directory(directory)

        # Sort heights for consistent plotting order
        heights = sorted(data_per_height.keys())

        config = fit_coil_temperature_model(data_per_height, mcu.get_coil_reference())
        model = CoilTemperatureCompensationModel(config, mcu)

        fig, axes = plt.subplots(2, len(heights), figsize=(25, 15))

        for i, height in enumerate(heights):
            samples = data_per_height[height]
            plot_samples(axes[0, i], samples, f"Height {height:.1f}mm", model)

        plot_all_samples(axes[1, 0], data_per_height, model)
        plot_compensation_magnitude(axes[1, 1], data_per_height, model)
        analyze_model(model, data_per_height)

        print(config)

        plt.tight_layout()
        plt.savefig(os.path.join(directory, "temperature_compensation_analysis.png"))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
