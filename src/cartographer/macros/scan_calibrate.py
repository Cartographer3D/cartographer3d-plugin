from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, final

from typing_extensions import assert_never, override

from cartographer.interfaces.printer import Macro, MacroParams, Position, Toolhead
from cartographer.macros.utils import get_enum_choice
from cartographer.probe import Probe, ScanModel

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration

    pass

logger = logging.getLogger(__name__)

DEFAULT_SCAN_MODEL_NAME = "default"


class ScanCalibrateMethod(Enum):
    TOUCH = "touch"
    MANUAL = "manual"


@final
class ScanCalibrateMacro(Macro):
    description = "Run the scan calibration"

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        config: Configuration,
    ) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        name = params.get("MODEL", DEFAULT_SCAN_MODEL_NAME).lower()
        method = get_enum_choice(params, "METHOD", ScanCalibrateMethod, default=ScanCalibrateMethod.MANUAL)

        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y"):
            msg = "Must home x and y before calibration"
            raise RuntimeError(msg)

        self._toolhead.move(
            x=self._config.bed_mesh.zero_reference_position[0],
            y=self._config.bed_mesh.zero_reference_position[1],
            speed=self._config.general.travel_speed,
        )
        self._toolhead.wait_moves()

        if method == ScanCalibrateMethod.TOUCH:
            return self._run_touch(name)
        elif method == ScanCalibrateMethod.MANUAL:
            return self._run_manual(name)

        assert_never(method)

    def _run_touch(self, name: str) -> None:
        trigger_pos = self._probe.perform_touch()
        pos = self._toolhead.get_position()
        self._toolhead.set_z_position(pos.z - trigger_pos)
        self._calibrate(name)

    def _run_manual(self, name: str) -> None:
        _, z_max = self._toolhead.get_axis_limits("z")
        self._toolhead.set_z_position(z=z_max - 10)

        logger.info("Triggering manual probe, please bring nozzle to 0.1mm above the bed")

        self._toolhead.manual_probe(partial(self._handle_manual_probe, name))

    def _handle_manual_probe(self, name: str, pos: Position | None) -> None:
        if pos is None:
            self._toolhead.clear_z_homing_state()
            return

        # TODO: Should this nozzle offset be customizable?
        # We assume the user will move the nozzle to 0.1mm above the bed
        self._toolhead.set_z_position(0.1)

        self._calibrate(name)

    def _calibrate(self, name: str):
        self._toolhead.move(z=5.5, speed=5)
        self._toolhead.wait_moves()

        with self._probe.scan.start_session() as session:
            session.wait_for(lambda samples: len(samples) > 50)
            self._toolhead.dwell(0.250)
            self._toolhead.move(z=0.1, speed=1)
            self._toolhead.dwell(0.250)
            self._toolhead.wait_moves()
            time = self._toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 50)
        self._toolhead.move(z=5, speed=5)

        samples = session.get_items()
        logger.debug("Collected %d samples", len(samples))

        model = ScanModel.fit(name, samples, z_offset=0)
        logger.debug("Calibration complete")

        self._config.save_scan_model(model)
        self._probe.scan.load_model(model.name)
        logger.info(
            "Scan model '%s' has been saved for the current session.\n"
            "The SAVE_CONFIG command will update the printer config file and restart the printer.",
            name,
        )
