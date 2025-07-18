from __future__ import annotations

from typing import Iterable, Literal, Protocol, TypeVar

from cartographer.interfaces.configuration import (
    BedMeshConfig,
    GeneralConfig,
    ScanConfig,
    ScanModelConfiguration,
    TouchConfig,
    TouchModelConfiguration,
)

K = TypeVar("K", bound=str)


def get_choice(params: ParseConfigWrapper, option: str, choices: Iterable[K], default: K) -> K:
    choice = params.get_str(option, default=default)
    choice_str = choice.lower()

    for k in choices:
        if k.lower() == choice_str:
            return k

    valid_choices = ", ".join(f"'{k.lower()}'" for k in choices)
    msg = f"Invalid choice '{choice}' for option '{option}'. Valid choices are: {valid_choices}"
    raise RuntimeError(msg)


class ParseConfigWrapper(Protocol):
    def get_name(self) -> str: ...
    def get_str(self, option: str, default: str) -> str: ...
    def get_optional_str(self, option: str) -> str | None: ...
    def get_float(
        self, option: str, default: float, minimum: float | None = None, maximum: float | None = None
    ) -> float: ...
    def get_required_float(self, option: str) -> float: ...
    def get_required_float_list(self, option: str, count: int | None = None) -> list[float]: ...
    def get_int(self, option: str, default: int) -> int: ...
    def get_required_int_list(self, option: str, count: int | None = None) -> list[int]: ...
    def get_bool(self, option: str, default: bool) -> bool: ...


T = TypeVar("T")


def list_to_tuple(lst: list[T]) -> tuple[T, T]:
    if len(lst) != 2:
        msg = f"Expected a list of length 2, got {len(lst)}"
        raise ValueError(msg)
    return (lst[0], lst[1])


def parse_general_config(wrapper: ParseConfigWrapper) -> GeneralConfig:
    return GeneralConfig(
        x_offset=wrapper.get_required_float("x_offset"),
        y_offset=wrapper.get_required_float("y_offset"),
        z_backlash=wrapper.get_float("z_backlash", default=0.05, minimum=0),
        travel_speed=wrapper.get_float("travel_speed", default=50, minimum=1),
        macro_prefix=wrapper.get_optional_str("macro_prefix"),
        verbose=wrapper.get_bool("verbose", default=False),
    )


_directions: list[Literal["x", "y"]] = ["x", "y"]
_paths: list[Literal["snake", "alternating_snake", "spiral", "random"]] = [
    "snake",
    "alternating_snake",
    "spiral",
    "random",
]


def parse_scan_config(wrapper: ParseConfigWrapper, models: dict[str, ScanModelConfiguration]) -> ScanConfig:
    return ScanConfig(
        samples=20,
        models=models,
        probe_speed=wrapper.get_float("probe_speed", default=5, minimum=0.1),
        mesh_runs=wrapper.get_int("mesh_runs", default=1),
        mesh_direction=get_choice(wrapper, "mesh_direction", _directions, default="x"),
        mesh_height=wrapper.get_float("mesh_height", default=4, minimum=1),
        mesh_path=get_choice(wrapper, "mesh_path", _paths, default="snake"),
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
        probe_count=list_to_tuple(wrapper.get_required_int_list("probe_count", count=2)),
        speed=wrapper.get_float("speed", default=50, minimum=1),
        horizontal_move_z=wrapper.get_float("horizontal_move_z", default=4, minimum=1),
        adaptive_margin=wrapper.get_float("adaptive_margin", default=5, minimum=0),
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
        z_offset=wrapper.get_float("z_offset", default=0, maximum=0),
    )
