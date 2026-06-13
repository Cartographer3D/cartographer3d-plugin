from __future__ import annotations


class ProbeTriggerError(RuntimeError):
    """Raised when probe triggers unexpectedly (e.g., before movement)."""


class PrinterShutdownError(RuntimeError):
    """Raised when an operation is interrupted because the printer entered shutdown."""

    def __init__(self) -> None:
        super().__init__("Printer entered shutdown")


class McuDisconnectedError(RuntimeError):
    """Raised when the MCU disconnects during an active operation."""

    def __init__(self) -> None:
        super().__init__("Cartographer MCU disconnected during session")
