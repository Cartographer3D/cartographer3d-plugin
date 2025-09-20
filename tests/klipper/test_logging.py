import logging
from typing import Callable

import pytest
from typing_extensions import TypeAlias

from cartographer.adapters.klipper.logging import GCodeConsoleFilter, GCodeConsoleFormatter, format_macro


@pytest.fixture
def formatter() -> GCodeConsoleFormatter:
    return GCodeConsoleFormatter()


LogRecordFactory: TypeAlias = "Callable[[int, str], logging.LogRecord]"


@pytest.fixture
def log_record() -> LogRecordFactory:
    def _make(level: int, msg: str, name: str = "test.logger"):
        return logging.LogRecord(name=name, level=level, pathname="test", lineno=1, msg=msg, args=(), exc_info=None)

    return _make


@pytest.mark.parametrize(
    "message,expected",
    [
        ("TOUCH", format_macro("TOUCH")),
        ("TOUCH_CALIBRATE THRESHOLD_START=1000", format_macro("TOUCH_CALIBRATE THRESHOLD_START=1000")),
        ("Start TOUCH then SAVE_CONFIG", f"Start {format_macro('TOUCH')} then {format_macro('SAVE_CONFIG')}"),
        ("Normal message", "Normal message"),
        (
            "Multiple TOUCH X=1 Y=2 SAVE_CONFIG",
            f"Multiple {format_macro('TOUCH X=1 Y=2')} {format_macro('SAVE_CONFIG')}",
        ),
        ("Ignore single capital C", "Ignore single capital C"),
    ],
)
def test_macro_formatting_info(
    formatter: GCodeConsoleFormatter, log_record: LogRecordFactory, message: str, expected: str
):
    record = log_record(logging.INFO, message)
    output = formatter.format(record)
    assert output == expected


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_formatting_no_prefix(formatter: GCodeConsoleFormatter, level: int, log_record: LogRecordFactory):
    record = log_record(level, "test message")
    output = formatter.format(record)
    assert output == "test message"


def test_formatting_error_prefix(formatter: GCodeConsoleFormatter, log_record: LogRecordFactory):
    record = log_record(logging.ERROR, "test error")
    output = formatter.format(record)
    assert output == "!! test error"


@pytest.mark.parametrize(
    "logger_name,level,expected",
    [
        ("adapters.klipper.mcu", logging.DEBUG, False),  # filtered out
        ("adapters.klipper.printer", logging.INFO, True),  # allowed
        ("macros.scan", logging.DEBUG, True),  # allowed
    ],
)
def test_console_filter_behavior(logger_name: str, level: int, expected: bool):
    record = logging.LogRecord(
        name=logger_name, level=level, pathname="test", lineno=1, msg="test", args=(), exc_info=None
    )
    filt = GCodeConsoleFilter()
    assert filt.filter(record) is expected
