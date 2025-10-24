from __future__ import annotations

import logging
import re
from typing import Protocol

from typing_extensions import override

module_name = __name__.split(".")[0]

root_logger = logging.getLogger(module_name)

LOG_PREFIX = "[cartographer] "


class Console(Protocol):
    def respond_raw(self, msg: str) -> None: ...


def setup_console_logger(console: Console) -> logging.Handler:
    """
    Configure console logging for the cartographer module.
    """
    # Remove any existing handlers of these types
    for handler in root_logger.handlers[:]:
        if isinstance(handler, (PrefixingHandler, GCodeConsoleHandler)):
            root_logger.removeHandler(handler)

    # Add a prefixing handler to root logger
    prefix_handler = PrefixingHandler()
    root_logger.addHandler(prefix_handler)

    console_handler = GCodeConsoleHandler(console)
    console_handler.setFormatter(GCodeConsoleFormatter())
    console_handler.addFilter(GCodeConsoleFilter())
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)  # Propagates all messages for the console handler

    return console_handler


MACRO_PATTERN = re.compile(r"\b([A-Z_]{2,}(?:\s+[A-Z0-9_]+=.*?)*)(?=\s|$)")


def format_macro(macro: str) -> str:
    """
    Format a macro string for display in the console.

    Parameters
    ----------
    macro : str
        The macro string to format.

    Returns
    -------
    str
        The formatted macro string wrapped in HTML anchor tags.
    """
    return f'<a class="command">{macro}</a>'


class PrefixingHandler(logging.Handler):
    """
    Handler that adds [cartographer] prefix to cartographer log records.
    It doesn't actually emit anything - just modifies the records.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setLevel(logging.NOTSET)  # Process all levels

    @override
    def emit(self, record: logging.LogRecord) -> None:
        record.msg = f"{LOG_PREFIX}{record.msg}"


class GCodeConsoleFormatter(logging.Formatter):
    """
    Formatter for console output that strips the [cartographer] prefix.

    This formatter removes the [cartographer] prefix added by PrefixingHandler,
    keeping console output clean while still applying macro highlighting.
    """

    def __init__(self) -> None:
        super().__init__("%(message)s")

    @override
    def format(self, record: logging.LogRecord) -> str:
        prefix = "!! " if record.levelno >= logging.ERROR else ""
        message = super().format(record)

        # Strip [cartographer] prefix if present
        if message.startswith(LOG_PREFIX):
            message = message[len(LOG_PREFIX) :]

        return prefix + MACRO_PATTERN.sub(lambda m: format_macro(m.group(0)), message)


class GCodeConsoleFilter(logging.Filter):
    """
    Filter to control which log records are sent to the console.
    """

    def __init__(self) -> None:
        super().__init__("%(message)s")

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return "klipper.mcu" not in record.name or record.levelno >= logging.WARNING


class GCodeConsoleHandler(logging.Handler):
    """
    Custom handler for sending log messages to the GCode console.

    Parameters
    ----------
    console : Console
        The console interface to send formatted messages to.
    """

    def __init__(self, console: Console) -> None:
        self.console: Console = console
        super().__init__()

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = self.format(record)
            self.console.respond_raw(f"{log_entry}\n")

        except Exception:
            self.handleError(record)
