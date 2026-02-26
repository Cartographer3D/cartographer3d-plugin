from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, Tuple

from typing_extensions import override

from cartographer.config.fields import option

if TYPE_CHECKING:
    from configfile import ConfigWrapper


class MeshDirection(str, Enum):
    X = "x"
    Y = "y"

    # Python 3.11+ changed str(StrEnum) to return "EnumClass.MEMBER" instead of the value.
    # Override to ensure str() always returns the raw value (e.g. "x", not "MeshDirection.X").
    @override
    def __str__(self) -> str:
        return self.value


class MeshPath(str, Enum):
    SNAKE = "snake"
    ALTERNATING_SNAKE = "alternating_snake"
    SPIRAL = "spiral"
    RANDOM = "random"

    # See MeshDirection.__str__ for rationale.
    @override
    def __str__(self) -> str:
        return self.value


def _list_to_tuple(lst: list[float]) -> tuple[float, float]:
    if len(lst) != 2:
        msg = f"Expected a list of length 2, got {len(lst)}"
        raise ValueError(msg)
    return (lst[0], lst[1])


def _list_to_int_tuple(lst: list[int]) -> tuple[int, int]:
    if len(lst) != 2:
        msg = f"Expected a list of length 2, got {len(lst)}"
        raise ValueError(msg)
    return (lst[0], lst[1])


def _parse_mesh_min(config: ConfigWrapper) -> tuple[float, float]:
    return _list_to_tuple(config.getfloatlist("mesh_min", count=2))


def _parse_mesh_max(config: ConfigWrapper) -> tuple[float, float]:
    return _list_to_tuple(config.getfloatlist("mesh_max", count=2))


def _parse_probe_count(config: ConfigWrapper) -> tuple[int, int]:
    return _list_to_int_tuple(config.getintlist("probe_count", count=2))


def _parse_zero_reference_position(config: ConfigWrapper) -> tuple[float, float]:
    return _list_to_tuple(config.getfloatlist("zero_reference_position", count=2))


Region = Tuple[Tuple[float, float], Tuple[float, float]]


def _parse_faulty_regions(config: ConfigWrapper) -> list[Region]:
    """Parse and validate faulty regions from config."""
    faulty_regions: list[Region] = []
    region_errors: list[str] = []

    for idx in range(1, 100):
        min_vals: list[float] | None = config.getfloatlist(f"faulty_region_{idx}_min", default=None, count=2)
        max_vals: list[float] | None = config.getfloatlist(f"faulty_region_{idx}_max", default=None, count=2)
        if min_vals is None or max_vals is None:
            continue

        min_tuple = _list_to_tuple(min_vals)
        max_tuple = _list_to_tuple(max_vals)
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


def _parse_coil_calibration(config: ConfigWrapper) -> CoilCalibrationConfiguration | None:
    calibration: list[float] | None = config.getfloatlist("calibration", count=4, default=None)
    if calibration is None:
        return None
    return CoilCalibrationConfiguration(
        a_a=calibration[0],
        a_b=calibration[1],
        b_a=calibration[2],
        b_b=calibration[3],
    )


def _parse_coil_max_temp(config: ConfigWrapper) -> float:
    min_temp = config.getfloat("min_temp", default=0, minval=-273.15)
    return config.getfloat("max_temp", default=105, minval=min_temp)


def _parse_model_name(config: ConfigWrapper) -> str:
    return config.get_name().split(" ")[-1]


def _parse_version_info(config: ConfigWrapper) -> ModelVersionInfo:
    software_version: str | None = config.get("software_version", default=None)
    mcu_version: str | None = config.get("mcu_version", default=None)
    if software_version is None:
        return ModelVersionInfo(mcu_version=mcu_version)
    return ModelVersionInfo(software_version=software_version, mcu_version=mcu_version)


def _parse_scan_domain(config: ConfigWrapper) -> tuple[float, float]:
    return _list_to_tuple(config.getfloatlist("domain", count=2))


@dataclass(frozen=True)
class GeneralConfig:
    mcu: str = option("The name of the Cartographer MCU, as defined by your [mcu cartographer] section.")
    x_offset: float = option("The distance (in mm) between the probe and the nozzle along the x-axis.")
    y_offset: float = option("The distance (in mm) between the probe and the nozzle along the y-axis.")
    z_backlash: float = option(default=0.05, min=0)
    travel_speed: float = option("The speed (in mm/s) of all travel moves.", default=50, min=1)
    lift_speed: float = option("The speed (in mm/s) of all Z lift moves.", default=5, min=1)
    verbose: bool = option("Set to yes to enable debug output.", default=False)
    macro_prefix: str | None = option(
        "A prefix to register a second set of Cartographer macros using, alongside 'CARTOGRAPHER_'."
        " E.g. 'CARTO' would result in 'CARTO_TOUCH_HOME'.",
        default=None,
    )


@dataclass(frozen=True)
class ScanConfig:
    samples: int = option(default=20)
    probe_speed: float = option("Speed (in mm/s) of the Z axis when probing.", default=5, min=0.1)
    mesh_runs: int = option(
        "Number of runs when doing a scan mesh. Consecutive runs will trace back the way it came from.",
        default=1,
    )
    mesh_height: float = option("The height (in mm) of the head when doing a mesh run.", default=3, min=1)
    mesh_direction: MeshDirection = option(
        "Axis of which to do the most moves along.",
        default=MeshDirection.X,
    )
    mesh_path: MeshPath = option(
        "The path to use when calibrating a scan mesh.",
        default=MeshPath.SNAKE,
    )
    models: dict[str, ScanModelConfiguration] = field(default_factory=dict)  # provided via override


@dataclass(frozen=True)
class TouchConfig:
    samples: int = option("The number of samples to use when doing a touch.", default=3, min=3)
    max_samples: int = option(
        "The maximum number of samples to do before giving up.",
        default=10,
    )
    max_touch_temperature: int = option(
        "The maximum allowed nozzle temperature to use when touching the plate."
        " This is set to 150C to avoid damaging your plate."
        " Change this AT YOUR OWN RISK.",
        default=150,
        key="UNSAFE_max_touch_temperature",
    )
    home_random_radius: float = option(
        "Radius around the Z home position to do touch.",
        default=0.0,
        min=0.0,
        key="EXPERIMENTAL_home_random_radius",
    )
    retract_distance: float = option("Retract distance (in mm) between touch samples.", default=2.0, min=1.0)
    sample_range: float = option("Acceptable range (in mm) between touch samples.", default=0.010, min=0.001, max=0.015)
    models: dict[str, TouchModelConfiguration] = field(default_factory=dict)  # provided via override


@dataclass(frozen=True)
class BedMeshConfig:
    mesh_min: tuple[float, float] = option(
        "Minimum coordinates of the mesh area.",
        parse_fn=_parse_mesh_min,
    )
    mesh_max: tuple[float, float] = option(
        "Maximum coordinates of the mesh area.",
        parse_fn=_parse_mesh_max,
    )
    probe_count: tuple[int, int] = option(
        "Number of probe points in X and Y.",
        parse_fn=_parse_probe_count,
    )
    zero_reference_position: tuple[float, float] = option(
        "Position used as zero reference for the mesh.",
        parse_fn=_parse_zero_reference_position,
    )
    faulty_regions: list[Region] = option(
        "Regions to exclude from mesh probing.",
        parse_fn=_parse_faulty_regions,
    )
    speed: float = option("Travel speed during mesh probing.", default=50, min=1)
    horizontal_move_z: float = option("Z height for horizontal moves during mesh.", default=5, min=1)
    adaptive_margin: float = option("Margin for adaptive mesh boundaries.", default=5, min=0)


@dataclass(frozen=True)
class ModelVersionInfo:
    """Version information for model compatibility checking."""

    mcu_version: str | None = None
    software_version: str = "1.0.2"


@dataclass(frozen=True)
class ScanModelConfiguration:
    name: str = option(parse_fn=_parse_model_name)
    coefficients: list[float] = option(
        parse_fn=lambda config: config.getfloatlist("coefficients"),
    )
    domain: tuple[float, float] = option(parse_fn=_parse_scan_domain)
    z_offset: float = option()
    reference_temperature: float = option()
    version_info: ModelVersionInfo = option(
        default=ModelVersionInfo(),
        parse_fn=_parse_version_info,
    )


@dataclass(frozen=True)
class TouchModelConfiguration:
    name: str = option(parse_fn=_parse_model_name)
    speed: float = option(min=1)
    z_offset: float = option(max=0)
    threshold: int = option(default=100)
    version_info: ModelVersionInfo = option(
        default=ModelVersionInfo(),
        parse_fn=_parse_version_info,
    )


@dataclass(frozen=True)
class CoilCalibrationConfiguration:
    a_a: float
    a_b: float
    b_a: float
    b_b: float


@dataclass(frozen=True)
class CoilConfiguration:
    name: str = option("The name of the coil temperature sensor.", default="cartographer_coil")
    min_temp: float = option(
        "The minimum temperature of the coil, below this will trigger a warning.",
        default=0,
        min=-273.15,
    )
    max_temp: float = option(
        "The maximum temperature of the coil, above this will trigger a warning.",
        default=105,
        parse_fn=_parse_coil_max_temp,
    )
    calibration: CoilCalibrationConfiguration | None = option(
        default=None,
        parse_fn=_parse_coil_calibration,
    )


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
