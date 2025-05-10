import logging
from typing import Literal, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Toolhead
from cartographer.probe.scan_mode import ScanMode

logger = logging.getLogger(__name__)


@final
class EstimateBacklashMacro(Macro):
    name = "ESTIMATE_BACKLASH"
    description = "Do a series of moves to estimate backlash on the Z axis."

    def __init__(self, toolhead: Toolhead, scan: ScanMode) -> None:
        self._scan = scan
        self._toolhead = toolhead

    @override
    def run(self, params: MacroParams) -> None:
        iterations = params.get_int("ITERATIONS", 10, minval=1)
        speed = 5
        height = 2
        delta = 0.2  # Smaller delta is better for detecting subtle backlash

        self._toolhead.move(z=height, speed=speed)
        samples: dict[Literal["up", "down"], list[float]] = {"up": [], "down": []}

        with self._scan.start_session():
            for _ in range(iterations):
                for direction in samples:
                    # When moving up, approach from below
                    dir = -1 if direction == "up" else 1
                    self._toolhead.move(z=height + delta * dir, speed=speed)
                    self._toolhead.move(z=height, speed=speed)
                    self._toolhead.wait_moves()
                    dist = self._scan.measure_distance()
                    samples[direction].append(dist)

        global_mean = np.mean(samples["up"] + samples["down"])
        mean_up = np.mean(samples["up"]) - global_mean
        mean_down = np.mean(samples["down"]) - global_mean
        std_up = np.std(samples["up"])
        std_down = np.std(samples["down"])
        backlash = mean_down - mean_up  # Positive = down sits lower than up

        logger.info(
            """
            Backlash estimation results over %d iterations:
            Mean moving upwards: %.5f mm
            Mean moving down: %.5f mm
            Std dev moving upwards: %.5f mm
            Std dev moving downwards: %.5f mm
            Estimated backlash: %.5f mm
            """,
            iterations,
            mean_up,
            mean_down,
            std_up,
            std_down,
            backlash,
        )
        if backlash < 0:
            logger.warning(
                """
                Backlash is negative, which is unexpected.
                This means the position after moving UP was measured as higher (or further)
                than the position after moving DOWN.
                Please check your printer's mechanical components (e.g., for slop, binding).
                """
            )
