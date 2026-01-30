from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class GeneralConfig:
    x_offset: float
    y_offset: float
    z_backlash: float
    travel_speed: float
    lift_speed: float
    verbose: bool
    macro_prefix: str | None


@dataclass(frozen=True)
class ScanConfig:
    samples: int
    models: dict[str, ScanModelConfiguration]
    probe_speed: float
    mesh_runs: int
    mesh_height: float
    mesh_direction: Literal["x", "y"]
    mesh_path: Literal["snake", "alternating_snake", "spiral", "random"]


@dataclass(frozen=True)
class TouchConfig:
    samples: int
    max_samples: int
    max_touch_temperature: int
    models: dict[str, TouchModelConfiguration]
    home_random_radius: float
    retract_distance: float


@dataclass(frozen=True)
class BedMeshConfig:
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    probe_count: tuple[int, int]
    speed: float
    horizontal_move_z: float
    adaptive_margin: float
    zero_reference_position: tuple[float, float]
    faulty_regions: list[tuple[tuple[float, float], tuple[float, float]]]


@dataclass(frozen=True)
class ModelVersionInfo:
    """Version information for model compatibility checking."""

    mcu_version: str | None = None
    software_version: str = "1.0.2"


@dataclass(frozen=True)
class ScanModelConfiguration:
    name: str
    coefficients: list[float]
    domain: tuple[float, float]
    z_offset: float
    reference_temperature: float
    version_info: ModelVersionInfo = ModelVersionInfo()


@dataclass(frozen=True)
class TouchModelConfiguration:
    name: str
    threshold: int
    speed: float
    z_offset: float
    version_info: ModelVersionInfo = ModelVersionInfo()


@dataclass(frozen=True)
class CoilCalibrationConfiguration:
    a_a: float
    a_b: float
    b_a: float
    b_b: float


@dataclass(frozen=True)
class CoilConfiguration:
    name: str
    min_temp: float
    max_temp: float
    calibration: CoilCalibrationConfiguration | None


class Configuration(Protocol):
    general: GeneralConfig
    scan: ScanConfig
    touch: TouchConfig
    bed_mesh: BedMeshConfig
    coil: CoilConfiguration

    def save_scan_model(self, config: ScanModelConfiguration) -> None: ...
    def save_touch_model(self, config: TouchModelConfiguration) -> None: ...
    def save_coil_model(self, config: CoilCalibrationConfiguration) -> None: ...
    def remove_scan_model(self, name: str) -> None: ...
    def remove_touch_model(self, name: str) -> None: ...
    def save_z_backlash(self, backlash: float) -> None: ...
    def log_runtime_warning(self, message: str) -> None: ...
