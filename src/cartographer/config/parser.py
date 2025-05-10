from __future__ import annotations

from typing import Protocol

from cartographer.interfaces.configuration import (
    BedMeshConfig,
    GeneralConfig,
    ScanConfig,
    ScanModelConfiguration,
    TouchConfig,
    TouchModelConfiguration,
)


class ParseConfigWrapper(Protocol):
    def get_name(self) -> str: ...
    def get_float(self, option: str, default: float, minimum: float | None = None) -> float: ...
    def get_required_float(self, option: str) -> float: ...
    def get_required_float_list(self, option: str, count: int | None = None) -> list[float]: ...
    def get_int(self, option: str, default: int) -> int: ...
    def get_bool(self, option: str, default: bool) -> bool: ...


def list_to_tuple(lst: list[float]) -> tuple[float, float]:
    if len(lst) != 2:
        msg = f"expected a list of length 2, got {len(lst)}"
        raise ValueError(msg)
    return (lst[0], lst[1])


def parse_general_config(wrapper: ParseConfigWrapper) -> GeneralConfig:
    return GeneralConfig(
        x_offset=wrapper.get_required_float("x_offset"),
        y_offset=wrapper.get_required_float("y_offset"),
        backlash_compensation=wrapper.get_float("backlash_compensation", default=0.05),
        travel_speed=wrapper.get_float("travel_speed", default=50, minimum=1),
        verbose=wrapper.get_bool("verbose", default=False),
    )


def parse_scan_config(wrapper: ParseConfigWrapper, models: dict[str, ScanModelConfiguration]) -> ScanConfig:
    return ScanConfig(
        samples=20,
        mesh_runs=wrapper.get_int("mesh_runs", default=1),
        models=models,
    )


def parse_touch_config(wrapper: ParseConfigWrapper, models: dict[str, TouchModelConfiguration]) -> TouchConfig:
    samples = wrapper.get_int("samples", default=5)
    return TouchConfig(
        samples=samples,
        max_samples=wrapper.get_int("max_samples", default=samples * 2),
        models=models,
    )


def parse_bed_mesh_config(wrapper: ParseConfigWrapper) -> BedMeshConfig:
    return BedMeshConfig(
        mesh_min=list_to_tuple(wrapper.get_required_float_list("mesh_min", count=2)),
        mesh_max=list_to_tuple(wrapper.get_required_float_list("mesh_max", count=2)),
        speed=wrapper.get_float("speed", default=50, minimum=1),
        horizontal_move_z=wrapper.get_float("horizontal_move_z", default=4, minimum=1),
        zero_reference_position=list_to_tuple(wrapper.get_required_float_list("zero_reference_position", count=2)),
    )


def parse_scan_model_config(wrapper: ParseConfigWrapper) -> ScanModelConfiguration:
    return ScanModelConfiguration(
        name=wrapper.get_name(),
        coefficients=wrapper.get_required_float_list("coefficients"),
        domain=list_to_tuple(wrapper.get_required_float_list("domain", count=2)),
        z_offset=wrapper.get_float("z_offset", default=0),
    )


def parse_touch_model_config(wrapper: ParseConfigWrapper) -> TouchModelConfiguration:
    return TouchModelConfiguration(
        name=wrapper.get_name(),
        threshold=wrapper.get_int("threshold", default=100),
        speed=wrapper.get_float("speed", default=50, minimum=1),
        z_offset=wrapper.get_float("z_offset", default=0),
    )
