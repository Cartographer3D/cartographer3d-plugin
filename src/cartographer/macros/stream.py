from __future__ import annotations

import logging
import os
import tempfile
import time
from enum import Enum
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams, Mcu, Sample
from cartographer.macros.utils import get_enum_choice

if TYPE_CHECKING:
    from cartographer.stream import Session

logger = logging.getLogger(__name__)


class StreamAction(Enum):
    START = "start"
    STOP = "stop"
    CANCEL = "cancel"
    STATUS = "status"


@final
class StreamMacro(Macro):
    description = "Controls a data stream of the cartographer readings to a file."

    def __init__(self, mcu: Mcu):
        self._mcu = mcu
        self._active_session: Session[Sample] | None = None
        self._output_file: str | None = None

    @override
    def run(self, params: MacroParams) -> None:
        action = get_enum_choice(params, "ACTION", StreamAction, default=StreamAction.STATUS)

        if action is StreamAction.START:
            self._start_streaming(params)
        elif action is StreamAction.STOP:
            self._stop_streaming(params)
        elif action is StreamAction.CANCEL:
            self._cancel_streaming()
        elif action is StreamAction.STATUS:
            self._show_status()

    def _start_streaming(self, params: MacroParams) -> None:
        if self._active_session is not None:
            msg = "Stream is already active. Use CARTOGRAPHER_STREAM ACTION=STOP to stop current stream."
            raise RuntimeError(msg)

        # Generate and validate output file path
        output_file = params.get("FILE", None)
        if output_file is None:
            output_file = self._generate_default_filename()

        self._validate_output_path(output_file)
        self._output_file = output_file

        # Start the streaming session
        self._active_session = self._mcu.start_session()

        logger.info("Started data streaming session, will save to: %s", self._output_file)

    def _stop_streaming(self, params: MacroParams) -> None:
        if self._active_session is None:
            msg = "No active stream to stop."
            raise RuntimeError(msg)

        output_file = params.get("FILE", self._output_file)
        if output_file is None:
            msg = "Output file path is not set. Please specify FILE parameter."
            raise RuntimeError(msg)

        # Validate new output file if it's different
        if output_file != self._output_file:
            self._validate_output_path(output_file)

        self._active_session.__exit__(None, None, None)
        samples = self._active_session.get_items()
        sample_count = len(samples)

        self._write_samples_to_csv(samples, output_file)

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

    def _validate_output_path(self, output_file: str) -> None:
        """Validate that we can write to the output path."""
        # Check if parent directory exists and is writable
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as e:
                msg = f"Cannot create directory for output file {output_file}: {e}"
                raise RuntimeError(msg) from e

        # Test file writability by attempting to create/open it
        try:
            with open(output_file, "w") as f:
                _ = f.write("")  # Write empty content to test
            os.remove(output_file)
        except OSError as e:
            msg = f"Cannot write to output file {output_file}: {e}"
            raise RuntimeError(msg) from e

    def _write_samples_to_csv(self, samples: list[Sample], output_file: str) -> None:
        """Write all samples to CSV file."""
        with open(output_file, "w", newline="") as f:
            # Write CSV header
            _ = f.write("time,frequency,temperature,position_x,position_y,position_z\n")

            # Write all sample rows
            for sample in samples:
                pos_x = sample.position.x if sample.position else ""
                pos_y = sample.position.y if sample.position else ""
                pos_z = sample.position.z if sample.position else ""

                row = f"{sample.time},{sample.frequency},{sample.temperature},{pos_x},{pos_y},{pos_z}\n"
                _ = f.write(row)

    def _generate_default_filename(self) -> str:
        """Generate a default filename in a safe location."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"cartographer_stream_{timestamp}.csv"

        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, filename)

    def _cleanup(self) -> None:
        """Clean up session state."""
        self._active_session = None
        self._output_file = None
