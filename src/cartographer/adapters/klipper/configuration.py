from __future__ import annotations

from dataclasses import astuple, replace
from functools import partial
from math import inf
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer import __version__
from cartographer.config.parser import (
    ParseConfigWrapper,
    parse_bed_mesh_config,
    parse_coil_config,
    parse_general_config,
    parse_scan_config,
    parse_scan_model_config,
    parse_touch_config,
    parse_touch_model_config,
)
from cartographer.interfaces.configuration import (
    CoilCalibrationConfiguration,
    Configuration,
    ModelVersionInfo,
    ScanModelConfiguration,
    TouchModelConfiguration,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper

    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu


@final
class KlipperConfigWrapper(ParseConfigWrapper):
    def __init__(self, config: ConfigWrapper, section_prefix: str = "") -> None:
        self._config = config
        self._section_prefix = section_prefix

    @override
    def get_name(self) -> str:
        return self._config.get_name().split(" ")[-1]

    @override
    def get_str(self, option: str, default: str) -> str:
        return self._config.get(option, default=default)

    @override
    def get_optional_str(self, option: str) -> str | None:
        return self._config.get(option, default=None)

    @override
    def get_float(
        self, option: str, default: float, minimum: float | None = None, maximum: float | None = None
    ) -> float:
        return self._config.getfloat(option, default=default, minval=minimum, maxval=maximum)

    @override
    def get_required_float(self, option: str, minimum: float = -inf, maximum: float = inf) -> float:
        return self._config.getfloat(option, minval=minimum, maxval=maximum)

    @override
    def get_required_float_list(self, option: str, count: int | None = None) -> list[float]:
        return self._config.getfloatlist(option, count=count)

    @override
    def get_float_list(self, option: str, count: int | None = None) -> list[float] | None:
        return self._config.getfloatlist(option, count=count, default=None)

    @override
    def get_int(self, option: str, default: int, minimum: int | None = None) -> int:
        return self._config.getint(option, default=default, minval=minimum)

    @override
    def get_required_int_list(self, option: str, count: int | None = None) -> list[int]:
        return self._config.getintlist(option, count=count)

    @override
    def get_bool(self, option: str, default: bool) -> bool:
        return self._config.getboolean(option, default=default)


@final
class KlipperConfiguration(Configuration):
    def __init__(self, config: ConfigWrapper, mcu: KlipperCartographerMcu) -> None:
        self.wrapper = config
        self._mcu = mcu
        self._config = config.get_printer().lookup_object("configfile")

        self.name = config.get_name()

        self._validate_stepper_z()

        self.general = parse_general_config(KlipperConfigWrapper(config))
        self.coil = parse_coil_config(KlipperConfigWrapper(config.getsection("cartographer coil")))

        self.bed_mesh = parse_bed_mesh_config(KlipperConfigWrapper(config.getsection("bed_mesh")))

        self.scan_model_prefix = f"{self.name} scan_model"
        scan_models = {
            cfg.get_name(): parse_scan_model_config(cfg)
            for cfg in (KlipperConfigWrapper(wrapper) for wrapper in config.get_prefix_sections(self.scan_model_prefix))
        }
        self.scan = parse_scan_config(KlipperConfigWrapper(config.getsection(f"{self.name} scan")), scan_models)

        self.touch_model_prefix = f"{self.name} touch_model"
        touch_models = {
            cfg.get_name(): parse_touch_model_config(cfg)
            for cfg in (
                KlipperConfigWrapper(wrapper) for wrapper in config.get_prefix_sections(self.touch_model_prefix)
            )
        }
        self.touch = parse_touch_config(KlipperConfigWrapper(config.getsection(f"{self.name} touch")), touch_models)

    @override
    def save_scan_model(self, config: ScanModelConfiguration) -> None:
        save = partial(self._config.set, f"{self.scan_model_prefix} {config.name}")
        save("coefficients", ",".join(map(str, config.coefficients)))
        save("domain", ",".join(map(str, config.domain)))
        save("z_offset", round(config.z_offset, 3))
        save("reference_temperature", round(config.reference_temperature, 2))

        # Save version information
        sw_version = __version__
        mcu_version = self._mcu.get_mcu_version()
        save("software_version", sw_version)
        save("mcu_version", mcu_version)

        updated_config = replace(
            config,
            version_info=ModelVersionInfo(
                software_version=sw_version,
                mcu_version=mcu_version,
            ),
        )
        self.scan.models[config.name] = updated_config

    @override
    def remove_scan_model(self, name: str) -> None:
        self._config.remove_section(f"{self.scan_model_prefix} {name}")
        _ = self.scan.models.pop(name, None)

    @override
    def save_touch_model(self, config: TouchModelConfiguration) -> None:
        save = partial(self._config.set, f"{self.touch_model_prefix} {config.name}")
        save("threshold", config.threshold)
        save("speed", config.speed)
        save("z_offset", round(config.z_offset, 3))

        # Save version information
        sw_version = __version__
        mcu_version = self._mcu.get_mcu_version()
        save("software_version", sw_version)
        save("mcu_version", mcu_version)

        updated_config = replace(
            config,
            version_info=ModelVersionInfo(
                software_version=sw_version,
                mcu_version=mcu_version,
            ),
        )
        self.touch.models[config.name] = updated_config

    @override
    def remove_touch_model(self, name: str) -> None:
        self._config.remove_section(f"{self.touch_model_prefix} {name}")
        _ = self.touch.models.pop(name, None)

    @override
    def save_z_backlash(self, backlash: float) -> None:
        self._config.set(self.name, "z_backlash", round(backlash, 5))

    @override
    def save_coil_model(self, config: CoilCalibrationConfiguration) -> None:
        value = ",".join(map(str, astuple(config)))
        self._config.set(f"{self.name} coil", "calibration", value)

    @override
    def log_runtime_warning(self, message: str) -> None:
        return self._config.runtime_warning(message)

    def _validate_stepper_z(self) -> None:
        if not self.wrapper.has_section("stepper_z"):
            return
        stepper_z = self.wrapper.getsection("stepper_z")
        if stepper_z.get("endstop_pin", default=None) != "probe:z_virtual_endstop":
            return

        homing_retract_dist = stepper_z.getfloat("homing_retract_dist", default=None, note_valid=False)
        if homing_retract_dist is None or homing_retract_dist != 0:
            msg = "Option 'homing_retract_dist' in section 'stepper_z' must be set to 0"
            raise self.wrapper.error(msg)
