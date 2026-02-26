from __future__ import annotations

from dataclasses import astuple, replace
from functools import partial
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer import __version__
from cartographer.config.fields import get_option_name, parse
from cartographer.interfaces.configuration import (
    BedMeshConfig,
    CoilCalibrationConfiguration,
    CoilConfiguration,
    Configuration,
    GeneralConfig,
    ModelVersionInfo,
    ScanConfig,
    ScanModelConfiguration,
    TouchConfig,
    TouchModelConfiguration,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper

    from cartographer.adapters.klipper.mcu.mcu import KlipperCartographerMcu


@final
class KlipperConfiguration(Configuration):
    def __init__(self, config: ConfigWrapper, mcu: KlipperCartographerMcu, general: GeneralConfig) -> None:
        self.wrapper = config
        self._mcu = mcu
        self._config = config.get_printer().lookup_object("configfile")

        self.name = config.get_name()

        self._validate_stepper_z()

        self.general = general
        self.coil = parse(CoilConfiguration, config.getsection("cartographer coil"))

        self.bed_mesh = parse(BedMeshConfig, config.getsection("bed_mesh"))

        self.scan_model_prefix = f"{self.name} scan_model"
        scan_models = {
            wrapper.get_name().split(" ")[-1]: parse(ScanModelConfiguration, wrapper)
            for wrapper in config.get_prefix_sections(self.scan_model_prefix)
        }
        self.scan = parse(ScanConfig, config.getsection(f"{self.name} scan"), models=scan_models)

        self.touch_model_prefix = f"{self.name} touch_model"
        touch_models = {
            wrapper.get_name().split(" ")[-1]: parse(TouchModelConfiguration, wrapper)
            for wrapper in config.get_prefix_sections(self.touch_model_prefix)
        }
        self.touch = parse(TouchConfig, config.getsection(f"{self.name} touch"), models=touch_models)

    @override
    def save_scan_model(self, config: ScanModelConfiguration) -> None:
        save = partial(self._config.set, f"{self.scan_model_prefix} {config.name}")
        _key = partial(get_option_name, ScanModelConfiguration)
        save(_key("coefficients"), ",".join(map(str, config.coefficients)))
        save(_key("domain"), ",".join(map(str, config.domain)))
        save(_key("z_offset"), round(config.z_offset, 3))
        save(_key("reference_temperature"), round(config.reference_temperature, 2))

        # Version info fields are part of ModelVersionInfo, not individual option() fields
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
        _key = partial(get_option_name, TouchModelConfiguration)
        save(_key("threshold"), config.threshold)
        save(_key("speed"), config.speed)
        save(_key("z_offset"), round(config.z_offset, 3))

        # Version info fields are part of ModelVersionInfo, not individual option() fields
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
        self._config.set(self.name, get_option_name(GeneralConfig, "z_backlash"), round(backlash, 5))

    @override
    def save_coil_model(self, config: CoilCalibrationConfiguration) -> None:
        value = ",".join(map(str, astuple(config)))
        self._config.set(f"{self.name} coil", get_option_name(CoilConfiguration, "calibration"), value)

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
