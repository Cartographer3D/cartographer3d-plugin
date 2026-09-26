from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.printer import Position, Sample
from cartographer.macros.temperature_calibrate import TemperatureCalibrateMacro, collection_targets
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


def test_collection_targets_step_up_to_max() -> None:
    assert collection_targets(40, 70, 0) == [70]
    assert collection_targets(40, 70, 10) == [50, 60, 70]
    assert collection_targets(40, 65, 10) == [50, 60, 65]


def _run(mocker: MockerFixture, touch: Mock | None, **params: str) -> tuple[Mock, Mock, Mock]:
    toolhead = mocker.Mock()
    toolhead.get_position = mocker.Mock(return_value=Position(175, 175, 1.2))
    toolhead.get_axis_limits = mocker.Mock(return_value=(0, 300))
    config = mocker.Mock()
    config.bed_mesh.zero_reference_position = (175, 175)
    mcu = mocker.MagicMock()
    macro = TemperatureCalibrateMacro(mcu, toolhead, config, mocker.Mock(), mocker.Mock(), mocker.Mock(), touch=touch)
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    write = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")
    _ = mocker.patch("cartographer.macros.temperature_calibrate._write_touches")
    wait = mocker.patch.object(macro, "_wait_for_temperature")
    macro_params = MockParams()
    macro_params.params = {"MIN_TEMP": "40", "MAX_TEMP": "60", "BED_TEMP": "110", "INTERLEAVE": "0", **params}
    macro.run(macro_params)
    return toolhead, wait, write


def test_touch_step_rereferences_z_before_each_step(mocker: MockerFixture) -> None:
    touch = mocker.Mock()
    touch.perform_probe = mocker.Mock(return_value=0.03)

    toolhead, wait, _ = _run(mocker, touch, TOUCH_STEP="10")

    heating = [c.kwargs["target_temp"] for c in wait.call_args_list if c.kwargs["cooling"] is False]
    # per phase: reach 39, then 50 / 60 with a touch before each and one closing touch; 3 phases
    assert heating == [39, 50, 60] * 3
    assert touch.perform_probe.call_count == 9
    assert toolhead.set_z_position.call_args_list == [mocker.call(1.2 - 0.03)] * 9


def test_without_touch_step_never_touches(mocker: MockerFixture) -> None:
    touch = mocker.Mock()

    toolhead, wait, _ = _run(mocker, touch)

    heating = [c.kwargs["target_temp"] for c in wait.call_args_list if c.kwargs["cooling"] is False]
    assert heating == [39, 60] * 3
    touch.perform_probe.assert_not_called()
    toolhead.set_z_position.assert_not_called()


def test_touch_step_without_touch_probe_is_rejected(mocker: MockerFixture) -> None:
    with pytest.raises(RuntimeError, match="TOUCH_STEP"):
        _ = _run(mocker, None, TOUCH_STEP="5")


def test_failed_touch_retries_then_continues_without(mocker: MockerFixture) -> None:
    touch = mocker.Mock()
    touch.perform_probe = mocker.Mock(side_effect=RuntimeError("triggered prior to movement"))

    toolhead, wait, _ = _run(mocker, touch, TOUCH_STEP="10")

    heating = [c.kwargs["target_temp"] for c in wait.call_args_list if c.kwargs["cooling"] is False]
    assert heating == [39, 50, 60] * 3  # every phase still collects
    assert touch.perform_probe.call_count == 2 * 3  # one retry per phase, then it stops touching
    toolhead.set_z_position.assert_not_called()


def test_phase_data_is_written_when_the_phase_aborts(mocker: MockerFixture) -> None:
    from cartographer.macros.temperature_calibrate import TemperatureStallError

    toolhead = mocker.Mock()
    toolhead.get_position = mocker.Mock(return_value=Position(175, 175, 1.2))
    toolhead.get_axis_limits = mocker.Mock(return_value=(0, 300))
    toolhead.get_last_move_time = mocker.Mock(return_value=0.0)
    mcu = mocker.MagicMock()
    config = mocker.Mock()
    config.bed_mesh.zero_reference_position = (175, 175)
    macro = TemperatureCalibrateMacro(mcu, toolhead, config, mocker.Mock(), mocker.Mock(), mocker.Mock())
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    write = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")

    def wait(target_temp: float, cooling: bool, phase_start_time: float | None = None) -> None:
        _ = target_temp, phase_start_time
        if cooling:
            callback = mcu.register_callback.call_args.args[0]
            callback(mocker.Mock(time=-1.0))  # from before the move ended: dropped
            callback(mocker.Mock(time=1.0))
            callback(mocker.Mock(time=1.05))  # within SAMPLE_INTERVAL of the last kept: dropped
            callback(mocker.Mock(time=1.1))
            return
        msg = "stalled"
        raise TemperatureStallError(msg)

    _ = mocker.patch.object(macro, "_wait_for_temperature", side_effect=wait)
    macro_params = MockParams()
    macro_params.params = {"MIN_TEMP": "40", "MAX_TEMP": "60", "BED_TEMP": "110", "INTERLEAVE": "0"}

    with pytest.raises(TemperatureStallError):
        macro.run(macro_params)

    assert write.call_count == 1  # the cooldown samples of the aborted phase
    assert [sample.time for sample in write.call_args.args[0]] == [1.0, 1.1]


def test_stream_is_kept_open_for_the_whole_run(mocker: MockerFixture) -> None:
    # Callbacks alone do not make the MCU stream; without an open session the coil
    # temperature freezes at the last sample and every wait stalls.
    toolhead = mocker.Mock()
    toolhead.get_position = mocker.Mock(return_value=Position(175, 175, 1.2))
    toolhead.get_axis_limits = mocker.Mock(return_value=(0, 300))
    mcu = mocker.MagicMock()
    config = mocker.Mock()
    config.bed_mesh.zero_reference_position = (175, 175)
    macro = TemperatureCalibrateMacro(mcu, toolhead, config, mocker.Mock(), mocker.Mock(), mocker.Mock())
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    _ = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")
    session = mcu.start_session.return_value

    def wait(target_temp: float, cooling: bool, phase_start_time: float | None = None) -> None:
        _ = target_temp, cooling, phase_start_time
        session.__enter__.assert_called_once()
        session.__exit__.assert_not_called()

    _ = mocker.patch.object(macro, "_wait_for_temperature", side_effect=wait)
    macro_params = MockParams()
    macro_params.params = {"MIN_TEMP": "40", "MAX_TEMP": "60", "BED_TEMP": "110", "INTERLEAVE": "0"}
    macro.run(macro_params)

    start_condition = mcu.start_session.call_args.args[0]
    assert start_condition(mocker.Mock()) is False  # streams, but never stores samples
    session.__exit__.assert_called_once()


class _Rig:
    """Fake clock + coil: temperature rises at `rate` C/s; samples are delivered on every sleep."""

    def __init__(self, mocker: MockerFixture, rate: float) -> None:
        self.now: float = 0.0
        self.rate: float = rate
        self.callbacks: list[Callable[[Sample], None]] = []
        self.toolhead: Mock = mocker.Mock()
        self.toolhead.get_position = mocker.Mock(return_value=Position(175, 175, 1.0))
        self.toolhead.get_axis_limits = mocker.Mock(return_value=(0, 300))
        self.toolhead.get_last_move_time = mocker.Mock(side_effect=lambda: self.now)
        self.mcu: Mock = mocker.MagicMock()
        self.mcu.register_callback = mocker.Mock(side_effect=self.callbacks.append)
        self.mcu.unregister_callback = mocker.Mock(side_effect=self.callbacks.remove)
        self.mcu.get_last_sample = mocker.Mock(side_effect=lambda: self._sample())
        self.scheduler: Mock = mocker.Mock()
        self.scheduler.sleep = mocker.Mock(side_effect=self._sleep)
        fake_time = mocker.Mock()
        fake_time.monotonic = mocker.Mock(side_effect=lambda: self.now)
        _ = mocker.patch("cartographer.macros.temperature_calibrate.time", fake_time)

    def _sample(self) -> Sample:
        return Sample(frequency=3e6, time=self.now, position=None, temperature=self.temperature, raw_count=0)

    @property
    def temperature(self) -> float:
        return 39 + self.rate * self.now

    def _sleep(self, seconds: float) -> None:
        self.now += seconds
        for callback in list(self.callbacks):
            callback(self._sample())


def _run_interleaved(mocker: MockerFixture, rig: _Rig, touch: Mock | None, **params: str) -> tuple[Mock, Mock]:
    config = mocker.Mock()
    config.bed_mesh.zero_reference_position = (175, 175)
    macro = TemperatureCalibrateMacro(
        rig.mcu, rig.toolhead, config, mocker.Mock(), mocker.Mock(), rig.scheduler, touch=touch
    )
    _ = mocker.patch("cartographer.macros.temperature_calibrate.scipy_helpers.raise_if_curve_fit_unavailable")
    _ = mocker.patch("cartographer.macros.temperature_calibrate._write_touches")
    write = mocker.patch("cartographer.macros.temperature_calibrate.write_samples_to_csv")
    wait = mocker.patch.object(macro, "_wait_for_temperature")  # cooldown and the initial min-1 wait
    macro_params = MockParams()
    macro_params.params = {"MIN_TEMP": "40", "MAX_TEMP": "60", "BED_TEMP": "110", **params}  # interleave is the default
    macro.run(macro_params)
    return wait, write


def test_interleave_cycles_heights_in_one_ramp(mocker: MockerFixture) -> None:
    rig = _Rig(mocker, rate=0.2)  # 39 -> 60 C in 105 s
    touch = mocker.Mock()
    touch.perform_probe = mocker.Mock(return_value=0.01)

    wait, write = _run_interleaved(mocker, rig, touch, TOUCH_STEP="10", DWELL="3")

    assert [c.kwargs["cooling"] for c in wait.call_args_list] == [True, False]  # ONE cooldown
    heights = [c.kwargs["z"] for c in rig.toolhead.move.call_args_list if set(c.kwargs) == {"z", "speed"}]
    # cooling height twice (start + cooldown), ramp start at 1, touch returns to 1, then 1/2/3 cycles
    assert heights[:4] == [200, 200, 1, 1]
    assert heights[4:10] == [1, 2, 3, 1, 2, 3]
    assert touch.perform_probe.call_count == 3  # before 50 and 60, plus the closing touch
    written = {c.args[1].split("temp_calib_")[1].split("_2")[0]: c.args[0] for c in write.call_args_list}
    assert set(written) == {"h1mm", "h2mm", "h3mm"}
    counts = [len(written[k]) for k in ("h1mm", "h2mm", "h3mm")]
    assert all(n > 50 for n in counts)  # every height sampled across the whole ramp
    assert max(s.temperature for s in written["h3mm"]) > 55


def test_interleave_aborts_on_stall(mocker: MockerFixture) -> None:
    from cartographer.macros.temperature_calibrate import TemperatureStallError

    rig = _Rig(mocker, rate=0.0)
    with pytest.raises(TemperatureStallError):
        _ = _run_interleaved(mocker, rig, None)
    assert rig.now >= 300
