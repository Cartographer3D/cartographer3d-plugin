from __future__ import annotations

from typing import Iterable, Literal, Protocol, TypeVar

from typing_extensions import TypeAlias

from cartographer.interfaces.configuration import (
    BedMeshConfig,
    CoilCalibrationConfiguration,
    CoilConfiguration,
    GeneralConfig,
    ModelVersionInfo,
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
    def get_required_float(self, option: str, minimum: float = ..., maximum: float = ...) -> float: ...
    def get_required_float_list(self, option: str, count: int | None = None) -> list[float]: ...
    def get_float_list(self, option: str, count: int | None = None) -> list[float] | None: ...
    def get_int(self, option: str, default: int, minimum: int | None = None) -> int: ...
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
        lift_speed=wrapper.get_float("lift_speed", default=5, minimum=1),
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
        mesh_height=wrapper.get_float("mesh_height", default=3, minimum=1),
        mesh_path=get_choice(wrapper, "mesh_path", _paths, default="snake"),
    )


def parse_touch_config(wrapper: ParseConfigWrapper, models: dict[str, TouchModelConfiguration]) -> TouchConfig:
    samples = wrapper.get_int("samples", default=3, minimum=3)
    return TouchConfig(
        samples=samples,
        max_samples=wrapper.get_int("max_samples", default=max(10, samples * 2)),
        max_touch_temperature=wrapper.get_int("UNSAFE_max_touch_temperature", default=150),
        home_random_radius=wrapper.get_float("EXPERIMENTAL_home_random_radius", default=0.0, minimum=0.0),
        models=models,
        retract_distance=wrapper.get_float("retract_distance", default=2.0, minimum=1.0),
    )


Region: TypeAlias = "tuple[tuple[float, float], tuple[float, float]]"


def _parse_faulty_regions(wrapper: ParseConfigWrapper) -> list[Region]:
    """Parse and validate faulty regions from config."""
    faulty_regions: list[Region] = []
    region_errors: list[str] = []

    for idx in range(1, 100):
        min_vals = wrapper.get_float_list(f"faulty_region_{idx}_min", None)
        max_vals = wrapper.get_float_list(f"faulty_region_{idx}_max", None)
        if min_vals is None or max_vals is None:
            continue

        min_tuple = list_to_tuple(min_vals)
        max_tuple = list_to_tuple(max_vals)
        errors = [
            f"faulty_region_{idx}: min[{axis}]={min_v} > max[{axis}]={max_v}"
            for axis, (min_v, max_v) in enumerate(zip(min_tuple, max_tuple))
            if min_v > max_v
        ]
        region_errors.extend(errors)
        faulty_regions.append((min_tuple, max_tuple))

    if region_errors:
        msg = (
            f"Invalid region bounds detected: {'; '.join(region_errors)}. "
            "Please verify that all min values are less than or equal to their corresponding max values."
        )
        raise ValueError(msg)

    return faulty_regions


def parse_bed_mesh_config(wrapper: ParseConfigWrapper) -> BedMeshConfig:
    faulty_regions = _parse_faulty_regions(wrapper)

    return BedMeshConfig(
        mesh_min=list_to_tuple(wrapper.get_required_float_list("mesh_min", count=2)),
        mesh_max=list_to_tuple(wrapper.get_required_float_list("mesh_max", count=2)),
        probe_count=list_to_tuple(wrapper.get_required_int_list("probe_count", count=2)),
        speed=wrapper.get_float("speed", default=50, minimum=1),
        horizontal_move_z=wrapper.get_float("horizontal_move_z", default=5, minimum=1),
        adaptive_margin=wrapper.get_float("adaptive_margin", default=5, minimum=0),
        zero_reference_position=list_to_tuple(wrapper.get_required_float_list("zero_reference_position", count=2)),
        faulty_regions=faulty_regions,
    )


def _parse_version_info(wrapper: ParseConfigWrapper) -> ModelVersionInfo:
    """Parse version information from model config."""
    software_version = wrapper.get_optional_str("software_version")
    mcu_version = wrapper.get_optional_str("mcu_version")

    if software_version is None:
        return ModelVersionInfo(mcu_version=mcu_version)

    return ModelVersionInfo(
        software_version=software_version,
        mcu_version=mcu_version,
    )


def parse_scan_model_config(wrapper: ParseConfigWrapper) -> ScanModelConfiguration:
    return ScanModelConfiguration(
        name=wrapper.get_name(),
        coefficients=wrapper.get_required_float_list("coefficients"),
        domain=list_to_tuple(wrapper.get_required_float_list("domain", count=2)),
        z_offset=wrapper.get_required_float("z_offset"),
        reference_temperature=wrapper.get_required_float("reference_temperature"),
        version_info=_parse_version_info(wrapper),
    )


def parse_touch_model_config(wrapper: ParseConfigWrapper) -> TouchModelConfiguration:
    return TouchModelConfiguration(
        name=wrapper.get_name(),
        threshold=wrapper.get_int("threshold", default=100),
        speed=wrapper.get_required_float("speed", minimum=1),
        z_offset=wrapper.get_required_float("z_offset", maximum=0),
        version_info=_parse_version_info(wrapper),
    )


ABSOLUTE_ZERO_TEMP = -273.15  # Celsius
ARBITRARY_MAX_TEMP = 9999.0


def parse_coil_config(wrapper: ParseConfigWrapper) -> CoilConfiguration:
    min_temp = wrapper.get_float("min_temp", default=0, minimum=ABSOLUTE_ZERO_TEMP)
    calibration = wrapper.get_float_list("calibration", count=4)
    calibration_config = (
        CoilCalibrationConfiguration(
            a_a=calibration[0],
            a_b=calibration[1],
            b_a=calibration[2],
            b_b=calibration[3],
        )
        if calibration
        else None
    )

    return CoilConfiguration(
        name=wrapper.get_str("name", default="cartographer_coil"),
        min_temp=min_temp,
        max_temp=wrapper.get_float("max_temp", default=105, minimum=min_temp),
        calibration=calibration_config,
    )
