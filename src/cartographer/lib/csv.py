from __future__ import annotations

import os
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Sample


def write_samples_to_csv(samples: list[Sample], output_file: str) -> None:
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


def generate_filepath(label: str) -> str:
    """Generate a path to a file in a safe location."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"cartographer_{label}_{timestamp}.csv"

    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, filename)


def validate_output_path(output_file: str) -> None:
    """Validate that we can write to the output path."""
    # Check if parent directory exists and is writable
    output_file = resolve_filepath(output_file)
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


def resolve_filepath(path: str) -> str:
    """Expand ~ and environment variables in a file path."""
    return os.path.expandvars(os.path.expanduser(path))
