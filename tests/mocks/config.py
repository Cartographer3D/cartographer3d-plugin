from __future__ import annotations

from dataclasses import replace
from typing import final

from typing_extensions import override

from cartographer.interfaces.configuration import (
    BedMeshConfig,
    CoilCalibrationConfiguration,
    CoilConfiguration,
    Configuration,
    GeneralConfig,
    ScanConfig,
    ScanModelConfiguration,
    TouchConfig,
    TouchModelConfiguration,
)

default_general_config = GeneralConfig(
    x_offset=0.0,
    y_offset=0.0,
    lift_speed=5,
    travel_speed=300.0,
    z_backlash=0,
    macro_prefix="carto",
    verbose=False,
)
default_scan_config = ScanConfig(
    samples=20,
    models={},
    probe_speed=5.0,
    mesh_runs=1,
    mesh_direction="x",
    mesh_height=4.0,
    mesh_path="snake",
)
default_touch_config = TouchConfig(
    samples=5,
    max_samples=10,
    max_touch_temperature=150,
    home_random_radius=0.0,
    retract_distance=2.0,
    models={},
)
default_bed_mesh_config = BedMeshConfig(
    mesh_min=(0.0, 0.0),
    mesh_max=(200.0, 200.0),
    probe_count=(10, 10),
    speed=100,
    horizontal_move_z=3,
    adaptive_margin=2,
    zero_reference_position=(100, 100),
    faulty_regions=[],
)
default_coil_configuration = CoilConfiguration(
    name="cartographer_coil",
    min_temp=5,
    max_temp=105,
    calibration=None,
)


@final
class MockConfiguration(Configuration):
    def __init__(
        self,
        *,
        general: GeneralConfig = default_general_config,
        scan: ScanConfig = default_scan_config,
        touch: TouchConfig = default_touch_config,
        bed_mesh: BedMeshConfig = default_bed_mesh_config,
        coil: CoilConfiguration = default_coil_configuration,
    ):
        self.runtime_warnings: list[str] = []
        self.general = general
        self.scan = scan
        self.touch = touch
        self.bed_mesh = bed_mesh
        self.coil = coil

    @override
    def save_scan_model(self, config: ScanModelConfiguration) -> None:
        self.scan.models[config.name] = config

    @override
    def remove_scan_model(self, name: str) -> None:
        _ = self.scan.models.pop(name, None)

    @override
    def save_touch_model(self, config: TouchModelConfiguration) -> None:
        self.touch.models[config.name] = config

    @override
    def remove_touch_model(self, name: str) -> None:
        _ = self.touch.models.pop(name, None)

    @override
    def save_z_backlash(self, backlash: float) -> None:
        self.general = replace(self.general, z_backlash=backlash)

    @override
    def save_coil_model(self, config: CoilCalibrationConfiguration) -> None:
        self.coil = replace(self.coil, calibration=config)

    @override
    def log_runtime_warning(self, message: str) -> None:
        self.runtime_warnings.append(message)
