from __future__ import annotations

import logging
from dataclasses import asdict
from functools import cached_property
from typing import TYPE_CHECKING, Callable, TypedDict, final

import mcu
from mcu import MCU_trsync
from mcu import TriggerDispatch as KlipperTriggerDispatch
from typing_extensions import override

from cartographer.adapters.klipper.mcu.async_processor import AsyncProcessor
from cartographer.adapters.klipper.mcu.commands import (
    HomeCommand,
    KlipperCartographerCommands,
    ThresholdCommand,
    TriggerMethod,
)
from cartographer.adapters.klipper.mcu.constants import (
    FREQUENCY_RANGE_PERCENT,
    INVALID_FREQUENCY_COUNTS,
    SENSOR_READY_TIMEOUT,
    SHORTED_FREQUENCY_VALUE,
    TRIGGER_HYSTERESIS,
    KlipperCartographerConstants,
)
from cartographer.adapters.klipper.mcu.stream import KlipperStream, KlipperStreamMcu
from cartographer.interfaces.printer import CoilCalibrationReference, Mcu, Position, Sample

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from reactor import Reactor, ReactorCompletion

    from cartographer.interfaces.multiprocessing import Scheduler
    from cartographer.stream import Session

logger = logging.getLogger(__name__)


class _RawData(TypedDict):
    clock: int
    data: int
    temp: int


@final
class KlipperCartographerMcu(Mcu, KlipperStreamMcu):
    _constants: KlipperCartographerConstants | None = None
    _commands: KlipperCartographerCommands | None = None

    @property
    def constants(self) -> KlipperCartographerConstants:
        if self._constants is None:
            msg = "Mcu not initialized"
            raise RuntimeError(msg)
        return self._constants

    @override
    def get_coil_reference(self) -> CoilCalibrationReference:
        return CoilCalibrationReference(
            min_frequency=self.constants.count_to_frequency(self.constants.minimum_count),
            min_frequency_temperature=self.constants.calculate_temperature(self.constants.minimum_adc_count),
        )

    @property
    def commands(self) -> KlipperCartographerCommands:
        if self._commands is None:
            msg = "Mcu not initialized"
            raise RuntimeError(msg)
        return self._commands

    @cached_property
    def kinematics(self):
        return self.printer.lookup_object("toolhead").get_kinematics()

    def __init__(
        self,
        config: ConfigWrapper,
        scheduler: Scheduler,
    ):
        self._sensor_ready = False
        self.printer = config.get_printer()
        self.klipper_mcu = mcu.get_printer_mcu(self.printer, config.get("mcu"))
        self._reactor: Reactor = self.klipper_mcu.get_printer().get_reactor()
        self._stream = KlipperStream[Sample](self, self._reactor)
        self.dispatch = KlipperTriggerDispatch(self.klipper_mcu)
        self._scheduler = scheduler

        self.motion_report = self.printer.load_object(config, "motion_report")

        # Async processor for handling raw data on main thread
        self._async_processor = AsyncProcessor[_RawData](
            self._reactor,
            self._process_raw_data,
        )

        self.printer.register_event_handler("klippy:mcu_identify", self._handle_mcu_identify)
        self.printer.register_event_handler("klippy:connect", self._handle_connect)
        self.printer.register_event_handler("klippy:shutdown", self._handle_shutdown)
        self.klipper_mcu.register_config_callback(self._initialize)

    @override
    def get_status(self, eventtime: float) -> dict[str, object]:
        return {
            "last_sample": asdict(self._stream.last_item) if self._stream.last_item else None,
            "constants": self._constants.get_status() if self._constants else None,
        }

    @override
    def get_last_sample(self) -> Sample | None:
        return self._stream.last_item

    def _initialize(self) -> None:
        self._constants = KlipperCartographerConstants(self.klipper_mcu)
        self._commands = KlipperCartographerCommands(self.klipper_mcu)
        self.klipper_mcu.register_response(self._handle_data, "cartographer_data")
        logger.info("Initialized %s MCU", self.klipper_mcu.get_status()["mcu_version"])

    @override
    def get_mcu_version(self) -> str:
        return self.klipper_mcu.get_status()["mcu_version"]

    @override
    def start_homing_scan(self, print_time: float, frequency: float) -> ReactorCompletion:
        self._ensure_sensor_ready()

        self._set_threshold(frequency)
        completion = self.dispatch.start(print_time)

        self.commands.send_home(
            HomeCommand(
                trsync_oid=self.dispatch.get_oid(),
                trigger_reason=MCU_trsync.REASON_ENDSTOP_HIT,
                trigger_invert=0,
                threshold=0,
                trigger_method=TriggerMethod.SCAN,
            )
        )
        return completion

    @override
    def start_homing_touch(self, print_time: float, threshold: int) -> ReactorCompletion:
        self._ensure_sensor_ready()

        completion = self.dispatch.start(print_time)

        self.commands.send_home(
            HomeCommand(
                trsync_oid=self.dispatch.get_oid(),
                trigger_reason=MCU_trsync.REASON_ENDSTOP_HIT,
                trigger_invert=0,
                threshold=threshold,
                trigger_method=TriggerMethod.TOUCH,
            )
        )
        return completion

    @override
    def stop_homing(self, home_end_time: float) -> float:
        self.dispatch.wait_end(home_end_time)
        self.commands.send_stop_home()
        result = self.dispatch.stop()
        if result >= MCU_trsync.REASON_COMMS_TIMEOUT:
            msg = "Communication timeout during homing"
            raise RuntimeError(msg)
        if result != MCU_trsync.REASON_ENDSTOP_HIT:
            return 0.0

        # TODO: Use a query state command for actual end time
        return home_end_time

    @override
    def start_session(self, start_condition: Callable[[Sample], bool] | None = None) -> Session[Sample]:
        return self._stream.start_session(start_condition)

    @override
    def register_callback(self, callback: Callable[[Sample], None]) -> None:
        return self._stream.register_callback(callback)

    @override
    def unregister_callback(self, callback: Callable[[Sample], None]) -> None:
        return self._stream.unregister_callback(callback)

    @override
    def start_streaming(self) -> None:
        self.commands.send_stream_state(enable=True)

    @override
    def stop_streaming(self) -> None:
        self.commands.send_stream_state(enable=False)

    @override
    def get_current_time(self) -> float:
        return self.printer.get_reactor().monotonic()

    def _set_threshold(self, trigger_frequency: float) -> None:
        trigger = self.constants.frequency_to_count(trigger_frequency)
        untrigger = self.constants.frequency_to_count(trigger_frequency * (1 - TRIGGER_HYSTERESIS))

        self.commands.send_threshold(ThresholdCommand(trigger, untrigger))

    def _handle_mcu_identify(self) -> None:
        for stepper in self.kinematics.get_steppers():
            if stepper.is_active_axis("z"):
                self.dispatch.add_stepper(stepper)

    def _handle_connect(self) -> None:
        self.stop_streaming()

    def _handle_shutdown(self) -> None:
        self.stop_streaming()

    def _handle_data(self, data: _RawData) -> None:
        """
        Handle raw data from MCU response handler (runs on MCU thread).

        This method is called from the MCU response thread and should not
        access any state that requires thread synchronization. Instead, we
        validate and queue the data for processing on the main reactor thread.

        Parameters:
        -----------
        data : _RawData
            Raw data from the MCU.
        """
        self._validate_data(data)
        self._async_processor.queue_item(data)

    def _process_raw_data(self, data: _RawData) -> None:
        """
        Process raw MCU data on the main reactor thread.

        This method runs on the main thread where it's safe to access
        stepper positions and other shared state.

        Parameters:
        -----------
        data : _RawData
            Raw data from the MCU to process.
        """
        count = data["data"]
        clock = self.klipper_mcu.clock32_to_clock64(data["clock"])
        time = self.klipper_mcu.clock_to_print_time(clock)

        frequency = self.constants.count_to_frequency(count)
        temperature = self.constants.calculate_temperature(data["temp"])
        position = self.get_requested_position(time)

        sample = Sample(
            raw_count=count,
            time=time,
            frequency=frequency,
            temperature=temperature,
            position=position,
        )
        self._stream.add_item(sample)

    _data_error: str | None = None

    def _validate_data(self, data: _RawData) -> None:
        """
        Validate raw data from MCU (thread-safe operation).

        This validation only checks the frequency count and doesn't access
        any shared state, so it's safe to call from the MCU response thread.

        Parameters:
        -----------
        data : _RawData
            Raw data from the MCU to validate.
        """
        count = data["data"]
        error: str | None = None
        if count == SHORTED_FREQUENCY_VALUE:
            error = "coil is shorted or not connected."
        elif count > self.constants.minimum_count * FREQUENCY_RANGE_PERCENT:
            error = "coil frequency reading exceeded max expected value, received %(count)d"

        if self._data_error == error:
            return
        self._data_error = error

        if error is None:
            return

        logger.debug(error, {"count": count})
        if len(self._stream.sessions) > 0:
            self.klipper_mcu.get_printer().invoke_shutdown(error % {"count": count})

    def get_requested_position(self, time: float) -> Position | None:
        """
        Get the requested position at a given time.

        This method MUST be called from the main reactor thread as it
        accesses stepper positions which are not thread-safe.

        Parameters:
        -----------
        time : float
            The time to query the position at.

        Returns:
        --------
        Position | None
            The position at the given time, or None if unavailable.
        """
        kinematics = self.kinematics
        stepper_pos = {
            stepper.get_name(): stepper.mcu_to_commanded_position(stepper.get_past_mcu_position(time))
            for stepper in kinematics.get_steppers()
        }
        position = kinematics.calc_position(stepper_pos)
        return Position(x=position[0], y=position[1], z=position[2])

    def _ensure_sensor_ready(self, timeout: float = SENSOR_READY_TIMEOUT) -> None:
        """
        Wait for the LDC sensor to return valid data.

        On the current firmware, the sensor may return error codes like
        83887360 (0x05000100) while initializing. This method polls until
        valid data is received.
        """
        if self._sensor_ready:
            return

        if self._is_sensor_ready():
            self._sensor_ready = True
            return

        logger.debug("Cartographer sensor not ready, waiting for %.1f", timeout)
        if not self._scheduler.wait_until(
            self._is_sensor_ready,
            timeout=timeout,
            poll_interval=0.1,
        ):
            last_sample = self._stream.last_item
            count = int(last_sample.raw_count) if last_sample else 0
            msg = (
                f"Cartographer not ready after {timeout:.1f}s. "
                f"Last count: {count} (0x{count:08X}). "
                "Check coil connection and power."
            )
            raise RuntimeError(msg)

        self._sensor_ready = True
        logger.debug("Cartographer sensor ready")

    def _is_sensor_ready(self) -> bool:
        """Check if a frequency reading from the LDC sensor is valid."""
        sample = self._stream.last_item
        if sample is None:
            return False
        return sample.raw_count not in INVALID_FREQUENCY_COUNTS
