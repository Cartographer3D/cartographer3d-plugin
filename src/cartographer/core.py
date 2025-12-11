from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, final

from cartographer.coil.temperature_compensation import (
    CoilTemperatureCompensationModel,
)
from cartographer.config.model_validator import validate_and_remove_incompatible_models
from cartographer.macros.axis_twist_compensation import (
    AxisTwistCompensationMacro,
)
from cartographer.macros.backlash import EstimateBacklashMacro
from cartographer.macros.bed_mesh.scan_mesh import (
    BedMeshCalibrateConfiguration,
    BedMeshCalibrateMacro,
)
from cartographer.macros.message import MessageMacro
from cartographer.macros.model_manager import ScanModelManager, TouchModelManager
from cartographer.macros.probe import (
    ProbeAccuracyMacro,
    ProbeMacro,
    QueryProbeMacro,
    ZOffsetApplyProbeMacro,
)
from cartographer.macros.query import QueryMacro
from cartographer.macros.scan import ScanAccuracyMacro
from cartographer.macros.scan_calibrate import (
    DEFAULT_SCAN_MODEL_NAME,
    ScanCalibrateMacro,
)
from cartographer.macros.stream import StreamMacro
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro
from cartographer.macros.touch import (
    DEFAULT_TOUCH_MODEL_NAME,
    TouchAccuracyMacro,
    TouchCalibrateMacro,
    TouchHomeMacro,
    TouchProbeMacro,
)
from cartographer.probe.probe import Probe
from cartographer.probe.scan_mode import ScanMode, ScanModeConfiguration
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration
from cartographer.task_executor import MultiprocessingExecutor
from cartographer.toolhead import BacklashCompensatingToolhead

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Macro, Toolhead
    from cartographer.runtime.adapters import Adapters

logger = logging.getLogger(__name__)


@dataclass
class MacroRegistration:
    name: str
    macro: Macro


@final
class PrinterCartographer:
    def __init__(self, adapters: Adapters) -> None:
        self.mcu = adapters.mcu
        self.config = adapters.config
        self.scheduler = adapters.scheduler
        self.task_executor = MultiprocessingExecutor(self.scheduler)

        # Initialize toolhead with optional backlash compensation
        toolhead = self._create_toolhead(adapters.toolhead)

        # Initialize probe modes
        self.scan_mode = self._create_scan_mode(toolhead, adapters)
        self.touch_mode = TouchMode(self.mcu, toolhead, TouchModeConfiguration.from_config(self.config))

        # Create probe
        probe = Probe(self.scan_mode, self.touch_mode)

        # Store specific macros needed by integrators
        self.probe_macro = ProbeMacro(probe)
        self.query_probe_macro = QueryProbeMacro(probe)

        # Register all macros
        self.macros = self._create_macro_registrations(probe, toolhead, adapters)

    def ready_callback(self) -> None:
        validate_and_remove_incompatible_models(self.config, self.mcu.get_mcu_version())

        if DEFAULT_SCAN_MODEL_NAME in self.config.scan.models:
            self.scan_mode.load_model(DEFAULT_SCAN_MODEL_NAME)

        if DEFAULT_TOUCH_MODEL_NAME in self.config.touch.models:
            self.touch_mode.load_model(DEFAULT_TOUCH_MODEL_NAME)

    def _register_macro(self, name: str, macro: Macro, use_prefix: bool = True) -> list[MacroRegistration]:
        """Register a macro with optional prefixing."""
        if not use_prefix:
            return [MacroRegistration(name, macro)]

        registrations = [MacroRegistration(f"CARTOGRAPHER_{name}", macro)]

        prefix = self.config.general.macro_prefix
        if prefix is not None:
            formatted = prefix.rstrip("_").upper() + "_" if prefix else ""
            registrations.append(MacroRegistration(f"{formatted}{name}", macro))

        return registrations

    def _create_toolhead(self, toolhead: Toolhead) -> Toolhead:
        """Create toolhead with optional backlash compensation."""
        if self.config.general.z_backlash > 0:
            return BacklashCompensatingToolhead(toolhead, self.config.general.z_backlash)
        return toolhead

    def _create_scan_mode(self, toolhead: Toolhead, adapters: Adapters) -> ScanMode:
        """Initialize scan mode with optional model loading."""
        temperature_compensation = (
            CoilTemperatureCompensationModel(self.config.coil.calibration, adapters.mcu)
            if self.config.coil.calibration
            else None
        )

        scan_mode = ScanMode(
            self.mcu,
            toolhead,
            ScanModeConfiguration.from_config(self.config),
            temperature_compensation,
            adapters.axis_twist_compensation,
        )

        return scan_mode

    def _create_macro_registrations(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapters: Adapters,
    ) -> list[MacroRegistration]:
        """Create all macro registrations."""
        registrations: list[MacroRegistration] = []

        # Core probe macros
        registrations.extend(self._create_probe_macro_registrations(probe, toolhead))

        # Cartographer-specific macros
        registrations.extend(self._create_cartographer_macro_registrations(probe, toolhead, adapters))

        # Scan-related macros
        registrations.extend(self._create_scan_macro_registrations(probe, toolhead))

        # Touch-related macros
        registrations.extend(self._create_touch_macro_registrations(probe, toolhead))

        # Axis twist compensation
        registrations.extend(self._create_axis_twist_compensation_registration(probe, toolhead, adapters))

        # Legacy macro aliases
        registrations.extend(self._create_legacy_macro_registrations())

        return registrations

    def _create_probe_macro_registrations(self, probe: Probe, toolhead: Toolhead) -> list[MacroRegistration]:
        """Create standard probe macro registrations."""
        return list(
            chain.from_iterable(
                [
                    self._register_macro("PROBE", self.probe_macro, use_prefix=False),
                    self._register_macro(
                        "PROBE_ACCURACY",
                        ProbeAccuracyMacro(probe, toolhead),
                        use_prefix=False,
                    ),
                    self._register_macro(
                        "QUERY_PROBE",
                        self.query_probe_macro,
                        use_prefix=False,
                    ),
                    self._register_macro(
                        "Z_OFFSET_APPLY_PROBE",
                        ZOffsetApplyProbeMacro(probe, toolhead, self.config),
                        use_prefix=False,
                    ),
                ]
            )
        )

    def _create_cartographer_macro_registrations(
        self, probe: Probe, toolhead: Toolhead, adapters: Adapters
    ) -> list[MacroRegistration]:
        """Create Cartographer-specific macro registrations."""
        return list(
            chain.from_iterable(
                [
                    self._register_macro(
                        "QUERY",
                        QueryMacro(self.mcu, self.scan_mode, self.touch_mode),
                    ),
                    self._register_macro(
                        "BED_MESH_CALIBRATE",
                        BedMeshCalibrateMacro(
                            probe,
                            toolhead,
                            adapters.bed_mesh,
                            adapters.axis_twist_compensation,
                            self.task_executor,
                            BedMeshCalibrateConfiguration.from_config(self.config),
                        ),
                        use_prefix=False,
                    ),
                    self._register_macro("STREAM", StreamMacro(self.mcu)),
                    self._register_macro(
                        "TEMPERATURE_CALIBRATE",
                        TemperatureCalibrateMacro(
                            self.mcu,
                            toolhead,
                            self.config,
                            adapters.gcode,
                            self.task_executor,
                            self.scheduler,
                        ),
                    ),
                ]
            )
        )

    def _create_scan_macro_registrations(self, probe: Probe, toolhead: Toolhead) -> list[MacroRegistration]:
        """Create scan-related macro registrations."""
        return list(
            chain.from_iterable(
                [
                    self._register_macro(
                        "SCAN_CALIBRATE",
                        ScanCalibrateMacro(probe, toolhead, self.config),
                    ),
                    self._register_macro(
                        "SCAN_ACCURACY",
                        ScanAccuracyMacro(self.scan_mode, toolhead, self.mcu),
                    ),
                    self._register_macro(
                        "SCAN_MODEL",
                        ScanModelManager(self.scan_mode, self.config),
                    ),
                    self._register_macro(
                        "ESTIMATE_BACKLASH",
                        EstimateBacklashMacro(toolhead, self.scan_mode, self.config),
                    ),
                ]
            )
        )

    def _create_touch_macro_registrations(self, probe: Probe, toolhead: Toolhead) -> list[MacroRegistration]:
        """Create touch-related macro registrations."""
        return list(
            chain.from_iterable(
                [
                    self._register_macro(
                        "TOUCH_CALIBRATE",
                        TouchCalibrateMacro(probe, self.mcu, toolhead, self.config, self.task_executor),
                    ),
                    self._register_macro(
                        "TOUCH_MODEL",
                        TouchModelManager(self.touch_mode, self.config),
                    ),
                    self._register_macro(
                        "TOUCH_PROBE",
                        TouchProbeMacro(self.touch_mode),
                    ),
                    self._register_macro(
                        "TOUCH_ACCURACY",
                        TouchAccuracyMacro(
                            self.touch_mode,
                            toolhead,
                            lift_speed=self.config.general.lift_speed,
                        ),
                    ),
                    self._register_macro(
                        "TOUCH_HOME",
                        TouchHomeMacro(
                            self.touch_mode,
                            toolhead,
                            lift_speed=self.config.general.lift_speed,
                            home_position=self.config.bed_mesh.zero_reference_position,
                            travel_speed=self.config.general.travel_speed,
                            random_radius=self.config.touch.home_random_radius,
                        ),
                    ),
                ]
            )
        )

    def _create_axis_twist_compensation_registration(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapters: Adapters,
    ) -> list[MacroRegistration]:
        """Create axis twist compensation macro registration."""
        if adapters.axis_twist_compensation:
            macro = AxisTwistCompensationMacro(
                probe,
                toolhead,
                adapters.axis_twist_compensation,
                self.config,
            )
        else:
            macro = MessageMacro(
                "Add [axis_twist_compensation] to your config to use CARTOGRAPHER_AXIS_TWIST_COMPENSATION."
            )

        return self._register_macro(
            "CARTOGRAPHER_AXIS_TWIST_COMPENSATION",
            macro,
            use_prefix=False,
        )

    def _create_legacy_macro_registrations(self) -> list[MacroRegistration]:
        """Create deprecation messages for renamed macros."""
        old_to_new = [
            ("TOUCH", "TOUCH_HOME"),
            ("CALIBRATE", "SCAN_CALIBRATE"),
            ("THRESHOLD_SCAN", "TOUCH_CALIBRATE"),
        ]

        return list(
            chain.from_iterable(
                self._register_macro(
                    old,
                    MessageMacro(f"Macro CARTOGRAPHER_{old} has been replaced by CARTOGRAPHER_{new}."),
                )
                for old, new in old_to_new
            )
        )

    def get_status(self, eventtime: float) -> dict[str, object]:
        """Get status information from all modes."""
        return {
            "scan": self.scan_mode.get_status(eventtime),
            "touch": self.touch_mode.get_status(eventtime),
            "mcu": self.mcu.get_status(eventtime),
        }
