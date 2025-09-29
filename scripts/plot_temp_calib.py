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

    print(f"Successfully loaded data for {len(data_per_height)} height(s): {sorted(data_per_height.keys())}mm")
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
        compensated_frequencies = normalize_frequencies(
            [model.compensate(sample.frequency, sample.temperature, 50) for sample in samples]
        )
        ax.scatter(temperatures, compensated_frequencies, alpha=0.6, color="tab:orange")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency vs Temperature")
    ax.grid(True, alpha=0.3)


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

        fig, axes = plt.subplots(1, len(heights) + 1, figsize=(15, 10))

        # Handle case where there's only one height (axes won't be an array)
        if len(heights) == 1:
            axes = [axes[0], axes[1]]

        for i, height in enumerate(heights):
            samples = data_per_height[height]
            plot_samples(axes[i], samples, f"Height {height:.1f}mm", model)

        plot_all_samples(axes[-1], data_per_height, model)
        print(config)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
