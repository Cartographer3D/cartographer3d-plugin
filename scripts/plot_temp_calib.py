# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import csv
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
    data_per_height: dict[float, list[Sample]] = {}
    heights = [1, 2, 3]
    for height in heights:
        data_per_height[float(height)] = read_samples_from_csv(
            f"./scripts/cartographer_tempcalib_height{height:d}mm.csv"
        )

    config = fit_coil_temperature_model(data_per_height, mcu.get_coil_reference())
    model = CoilTemperatureCompensationModel(config, mcu)

    fig, axes = plt.subplots(1, len(heights) + 1, figsize=(15, 10))
    for (height, samples), ax in zip(data_per_height.items(), axes):
        plot_samples(ax, samples, f"Height {height:.0f}mm", model)

    plot_all_samples(axes[-1], data_per_height, model)

    plt.tight_layout()
    plt.show()
