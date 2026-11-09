from __future__ import annotations

from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from cartographer.interfaces.printer import MacroParams
    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode


@final
class Probe:
    """Main probe class managing mode instances"""

    def __init__(self, scan: ScanMode, touch: TouchMode):
        self.scan = scan
        self.touch = touch

    def query_is_triggered(self) -> bool:
        return self.scan.query_is_triggered(0)

    def perform_scan(self, *, gcmd: MacroParams) -> float:
        return self.scan.perform_probe(gcmd=gcmd)

    def perform_touch(self, *, gcmd: MacroParams) -> float:
        return self.touch.perform_probe(gcmd=gcmd)
