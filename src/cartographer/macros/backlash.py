from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Toolhead
from cartographer.toolhead import BacklashCompensatingToolhead

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe.scan_mode import ScanMode

logger = logging.getLogger(__name__)


@final
class EstimateBacklashMacro(Macro):
    description = "Do a series of moves to estimate backlash on the Z axis."

    def __init__(self, toolhead: Toolhead, scan: ScanMode, config: Configuration) -> None:
        self._scan = scan
        self._toolhead = toolhead.toolhead if isinstance(toolhead, BacklashCompensatingToolhead) else toolhead
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        calibrate = params.get("CALIBRATE", None) is not None
        iterations = params.get_int("ITERATIONS", default=10, minval=1)
        delta = params.get_float("DELTA", default=0.2, minval=0.2, maxval=1)
        speed = 5
        height = 2

        self._toolhead.move(z=height, speed=speed)
        if calibrate:
            x, y = self._config.bed_mesh.zero_reference_position
            self._toolhead.move(x=x, y=y, speed=self._config.general.travel_speed)
        self._toolhead.wait_moves()

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
        backlash = float(mean_down - mean_up)  # Positive = down sits lower than up

        t_stat, df = welchs_ttest(samples["down"], samples["up"])

        logger.debug(
            "Backlash estimation details (%d iterations):\n"
            "Mean after moving up:     %.5f mm\n"
            "Mean after moving down:   %.5f mm\n"
            "Std dev (up):             %.5f mm\n"
            "Std dev (down):           %.5f mm\n"
            "Estimated raw backlash:   %.5f mm\n"
            "Welch's t-test: t=%.5f, df=%.2f",
            iterations,
            mean_up,
            mean_down,
            std_up,
            std_down,
            backlash,
            t_stat,
            df,
        )

        if abs(t_stat) < 2.0:
            backlash = 0.0
            logger.info(
                "Backlash test over %d samples found no significant difference (|t|=%.2f).",
                iterations,
                abs(t_stat),
            )
        else:
            logger.info(
                "Backlash test over %d samples detected a consistent %.3f mm difference.",
                iterations,
                backlash,
            )

        if backlash < 0:
            logger.warning(
                "Estimated backlash is negative (%.5f mm), which is unexpected.\n"
                "This suggests up moves measured higher than down moves."
                "Check for looseness or mechanical slop. Ignoring result.",
                backlash,
            )
            return

        if calibrate:
            self._config.save_z_backlash(backlash)
            if backlash == 0.0:
                logger.info(
                    "Backlash calibration complete: no compensation required.\n"
                    "Your printer appears mechanically tight.\n"
                    "The SAVE_CONFIG command will update your config and restart the printer."
                )
            else:
                logger.info(
                    "Backlash calibration complete: %.3f mm compensation will be used.\n"
                    "The SAVE_CONFIG command will update your config and restart the printer.",
                    backlash,
                )


def welchs_ttest(a_in: list[float], b_in: list[float]) -> tuple[float, float]:
    a: NDArray[np.float_] = np.asarray(a_in, dtype=float)
    b: NDArray[np.float_] = np.asarray(b_in, dtype=float)

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = len(a)
    n_b = len(b)

    t_stat = (mean_a - mean_b) / np.sqrt(var_a / n_a + var_b / n_b)

    # Degrees of freedom (Welch–Satterthwaite equation)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df = numerator / denominator

    return float(t_stat), float(df)
