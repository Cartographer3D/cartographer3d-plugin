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
