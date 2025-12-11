from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from cartographer.interfaces.printer import ProbeMode
    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode


@final
class Probe:
    """Main probe class managing mode instances"""

    def __init__(self, scan: ScanMode, touch: TouchMode):
        self.scan = scan
        self.touch = touch
        self.current_mode: ProbeMode = scan

    def query_is_triggered(self) -> bool:
        return self.scan.query_is_triggered(0)

    def perform_probe(self) -> float:
        return self.current_mode.perform_probe()

    def perform_scan(self) -> float:
        return self.scan.perform_probe()

    def perform_touch(self) -> float:
        return self.touch.perform_probe()

    @contextmanager
    def as_touch(self):
        orig_mode = self.current_mode
        try:
            self.current_mode = self.touch
            yield
        finally:
            self.current_mode = orig_mode
