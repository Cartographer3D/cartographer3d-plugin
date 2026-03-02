from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Mcu, Sample
from cartographer.lib.csv import generate_filepath, resolve_filepath, validate_output_path, write_samples_to_csv
from cartographer.macros.fields import param, parse

if TYPE_CHECKING:
    from cartographer.stream import Session

logger = logging.getLogger(__name__)


class StreamAction(Enum):
    START = "start"
    STOP = "stop"
    CANCEL = "cancel"
    STATUS = "status"


@dataclass(frozen=True)
class StreamParams:
    """Parameters for CARTOGRAPHER_STREAM."""

    action: StreamAction = param("Stream action", default=StreamAction.STATUS)
    file: str | None = param("Output file path", default=None)


@final
class StreamMacro(Macro):
    description = "Controls a data stream of the cartographer readings to a file."

    def __init__(self, mcu: Mcu):
        self._mcu = mcu
        self._active_session: Session[Sample] | None = None
        self._output_file: str | None = None

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(StreamParams, params)

        if p.action is StreamAction.START:
            self._start_streaming(p)
        elif p.action is StreamAction.STOP:
            self._stop_streaming(p)
        elif p.action is StreamAction.CANCEL:
            self._cancel_streaming()
        elif p.action is StreamAction.STATUS:
            self._show_status()

    def _start_streaming(self, p: StreamParams) -> None:
        if self._active_session is not None:
            msg = "Stream is already active. Use CARTOGRAPHER_STREAM ACTION=STOP to stop current stream."
            raise RuntimeError(msg)

        # Generate and validate output file path
        output_file = p.file or generate_filepath("stream")

        validate_output_path(output_file)
        self._output_file = output_file

        # Start the streaming session
        self._active_session = self._mcu.start_session()

        logger.info("Started data streaming session, will save to: %s", self._output_file)

    def _stop_streaming(self, p: StreamParams) -> None:
        if self._active_session is None:
            msg = "No active stream to stop."
            raise RuntimeError(msg)

        output_file = p.file or self._output_file
        if output_file is None:
            msg = "Output file path is not set. Please specify FILE parameter."
            raise RuntimeError(msg)
        output_file = resolve_filepath(output_file)

        # Validate new output file if it's different
        if output_file != self._output_file:
            validate_output_path(output_file)

        self._active_session.__exit__(None, None, None)
        samples = self._active_session.get_items()
        sample_count = len(samples)

        write_samples_to_csv(samples, output_file)

        logger.info("Stopped data streaming. Collected %d samples. File saved: %s", sample_count, output_file)
        self._cleanup()

    def _cancel_streaming(self) -> None:
        if self._active_session is None:
            msg = "No active stream to cancel."
            raise RuntimeError(msg)

        self._active_session.__exit__(None, None, None)
        logger.info("Cancelled data streaming session")
        self._cleanup()

    def _show_status(self) -> None:
        if self._active_session is None:
            logger.info("No active data stream. Use CARTOGRAPHER_STREAM ACTION=START to begin streaming.")
            return

        sample_count = len(self._active_session.items)
        logger.info("Active data stream: %d samples collected, will save to: %s", sample_count, self._output_file)

    def _cleanup(self) -> None:
        """Clean up session state."""
        self._active_session = None
        self._output_file = None
