from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict, final

from mcu import MCU_trsync
from typing_extensions import override

from cartographer.interfaces.printer import CoilCalibrationReference, Mcu, Sample
from cartographer.mcu.async_processor import AsyncProcessor
from cartographer.mcu.commands import (
    CartographerCommands,
    HomeCommand,
    ThresholdCommand,
    TriggerMethod,
)
from cartographer.mcu.constants import (
    FREQUENCY_RANGE_PERCENT,
    INVALID_FREQUENCY_COUNTS,
    SENSOR_READY_TIMEOUT,
    SHORTED_FREQUENCY_VALUE,
    TRIGGER_HYSTERESIS,
    CartographerConstants,
)
from cartographer.mcu.stream import CartographerStream, CartographerStreamMcu

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcu import MCU, TriggerDispatch
    from reactor import Reactor, ReactorCompletion

    from cartographer.interfaces.mcu_platform import McuPlatform
    from cartographer.interfaces.multiprocessing import Scheduler
    from cartographer.stream import Session

logger = logging.getLogger(__name__)


class _RawData(TypedDict):
    clock: int
    data: int
    temp: int


@final
class CartographerMcu(Mcu, CartographerStreamMcu):
    _constants: CartographerConstants | None = None
    _commands: CartographerCommands | None = None

    @property
    def constants(self) -> CartographerConstants:
        if self._constants is None:
            msg = "Cartographer MCU not initialized"
            raise RuntimeError(msg)
        return self._constants

    @override
    def get_coil_reference(self) -> CoilCalibrationReference:
        return CoilCalibrationReference(
            min_frequency=self.constants.count_to_frequency(self.constants.minimum_count),
            min_frequency_temperature=self.constants.calculate_temperature(self.constants.minimum_adc_count),
        )

    @property
    def commands(self) -> CartographerCommands:
        if self._commands is None:
            msg = "Cartographer MCU not initialized"
            raise RuntimeError(msg)
        if self._platform.is_disconnected():
            msg = "Cartographer MCU is disconnected"
            raise RuntimeError(msg)
        return self._commands

    def __init__(
        self,
        platform: McuPlatform,
        scheduler: Scheduler,
    ) -> None:
        self._platform: McuPlatform = platform
        self._sensor_ready: bool = False
        self._reconnect_callbacks: list[Callable[[], None]] = []
        reactor: Reactor = platform.get_reactor()
        self._stream: CartographerStream[Sample] = CartographerStream[Sample](self, reactor)
        self.dispatch: TriggerDispatch = platform.create_trigger_dispatch()
        self._scheduler: Scheduler = scheduler
        self._async_processor: AsyncProcessor[_RawData] = AsyncProcessor[_RawData](reactor, self._process_raw_data)
        platform.register_lifecycle_handlers(
            on_identify=self._handle_mcu_identify,
            on_connect=self._handle_connect,
            on_shutdown=self._handle_shutdown,
            on_reconnect=self._handle_reconnect,
            on_disconnect=self._handle_disconnect,
        )
        platform.register_config_callback(self._initialize)

    @property
    def host_mcu(self) -> MCU:
        return self._platform.host_mcu

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
        if self._constants is None:
            self._constants = CartographerConstants(self._platform)
        self._constants.initialize()
        if self._commands is None:
            self._commands = CartographerCommands(self._platform)
        self._commands.initialize()
        self._platform.register_data_response(self._handle_data, self._DATA_MSG_FORMAT, self._DATA_MSG_NAME)
        self._sensor_ready = False
        logger.info("Initialized %s MCU", self.get_mcu_version() or "unknown")

    _DATA_MSG_FORMAT = "cartographer_data clock=%u data=%u temp=%u"
    _DATA_MSG_NAME = "cartographer_data"

    @override
    def get_mcu_version(self) -> str | None:
        return self._platform.get_mcu_version()

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
        self._async_processor.set_immediate(True)

    @override
    def stop_streaming(self) -> None:
        self._async_processor.set_immediate(False)
        self.commands.send_stream_state(enable=False)

    @override
    def get_current_time(self) -> float:
        return self._platform.get_reactor_time()

    def _set_threshold(self, trigger_frequency: float) -> None:
        trigger = self.constants.frequency_to_count(trigger_frequency)
        untrigger = self.constants.frequency_to_count(trigger_frequency * (1 - TRIGGER_HYSTERESIS))

        self.commands.send_threshold(ThresholdCommand(trigger, untrigger))

    def _handle_mcu_identify(self) -> None:
        for stepper in self._platform.get_z_steppers():
            self._platform.add_stepper_to_dispatch(self.dispatch, stepper)

    def _handle_connect(self) -> None:
        if self._commands is not None and not self._platform.is_disconnected():
            self.stop_streaming()

    def _handle_shutdown(self) -> None:
        if self._commands is not None and not self._platform.is_disconnected():
            self.stop_streaming()

    def register_reconnect_callback(self, callback: Callable[[], None]) -> None:
        self._reconnect_callbacks.append(callback)

    def _handle_reconnect(self) -> None:
        logger.info("Cartographer MCU reconnected")
        for callback in self._reconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception("Error in reconnect callback")
                self._platform.invoke_shutdown(f"Cartographer MCU reconnect failed: {e}")
                return

    def _handle_disconnect(self) -> None:
        logger.warning("Cartographer MCU disconnected")
        self._sensor_ready = False
        self._stream.abort_all_sessions()

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
        clock = self._platform.clock32_to_clock64(data["clock"])
        time = self._platform.clock_to_print_time(clock)

        frequency = self.constants.count_to_frequency(count)
        temperature = self.constants.calculate_temperature(data["temp"])
        position = self._platform.get_requested_position(time)

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
        Validate raw data from MCU.

        Called from the MCU response thread. Accesses ``_constants`` for
        range validation; if constants are not yet initialized (early
        startup), range validation is skipped.

        Parameters:
        -----------
        data : _RawData
            Raw data from the MCU to validate.
        """
        count = data["data"]
        error: str | None = None
        if count == SHORTED_FREQUENCY_VALUE:
            error = "coil is shorted or not connected."
        else:
            constants = self._constants
            if constants is not None and count > constants.minimum_count * FREQUENCY_RANGE_PERCENT:
                error = "coil frequency reading exceeded max expected value, received %(count)d"

        if self._data_error == error:
            return
        self._data_error = error

        if error is None:
            return

        logger.debug(error, {"count": count})
        if len(self._stream.sessions) > 0:
            self._platform.invoke_shutdown(error % {"count": count})

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
