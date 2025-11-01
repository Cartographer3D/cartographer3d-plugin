from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.configuration import TouchModelConfiguration
from cartographer.interfaces.printer import Macro, MacroParams, Mcu
from cartographer.macros.touch_calibrate.frequency_analysis import FrequencyAnalysisTouchCalibrateMethod
from cartographer.macros.touch_calibrate.iterative import IterativeTouchCalibrateMethod
from cartographer.macros.utils import get_enum_choice

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.printer import Toolhead
    from cartographer.probe.probe import Probe


logger = logging.getLogger(__name__)

DEFAULT_TOUCH_MODEL_NAME = "default"
DEFAULT_Z_OFFSET = -0.05


class TouchCalibrateMethod(Enum):
    ITERATIVE = "iterative"
    FREQUENCY_ANALYSIS = "frequency_analysis"


@final
class TouchCalibrateMacro(Macro):
    description = "Run the touch calibration"

    def __init__(self, probe: Probe, mcu: Mcu, toolhead: Toolhead, config: Configuration) -> None:
        self._config = config
        self._probe = probe
        self.frequency_analysis = FrequencyAnalysisTouchCalibrateMethod(probe, mcu, toolhead, config)

        self.iterative = IterativeTouchCalibrateMethod(probe, mcu, toolhead, config)

    @override
    def run(self, params: MacroParams) -> None:
        method = get_enum_choice(params, "METHOD", TouchCalibrateMethod, default=TouchCalibrateMethod.ITERATIVE)
        name = params.get("MODEL", DEFAULT_TOUCH_MODEL_NAME).lower()
        if method is TouchCalibrateMethod.FREQUENCY_ANALYSIS:
            result = self.frequency_analysis.run(params)
        else:
            result = self.iterative.run(params)

        if result is None:
            return
        threshold, speed = result

        # Save the model
        model = TouchModelConfiguration(
            name=name,
            threshold=threshold,
            speed=speed,
            z_offset=DEFAULT_Z_OFFSET,
        )

        self._config.save_touch_model(model)
        self._probe.touch.load_model(name)

        logger.info(
            "Touch model '%s' has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.",
            name,
        )
