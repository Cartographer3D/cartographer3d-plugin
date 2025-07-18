from __future__ import annotations

from dataclasses import replace
from typing import final

from typing_extensions import override

from cartographer.interfaces.configuration import (
    BedMeshConfig,
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
    travel_speed=300.0,
    z_backlash=0,
    macro_prefix="cartographer",
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
)


@final
class MockConfiguration(Configuration):
    def __init__(
        self,
        *,
        general: GeneralConfig | None = None,
        scan: ScanConfig | None = None,
        touch: TouchConfig | None = None,
        bed_mesh: BedMeshConfig | None = None,
    ):
        self.general = general or default_general_config
        self.scan = scan or default_scan_config
        self.touch = touch or default_touch_config
        self.bed_mesh = bed_mesh or default_bed_mesh_config

    @override
    def save_scan_model(self, config: ScanModelConfiguration) -> None:
        self.scan.models[config.name] = config

    @override
    def save_touch_model(self, config: TouchModelConfiguration) -> None:
        self.touch.models[config.name] = config

    @override
    def save_z_backlash(self, backlash: float) -> None:
        self.general = replace(self.general, z_backlash=backlash)
