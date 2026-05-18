# Interface to Klipper micro-controller code
#
# Copyright (C) 2016-2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
# Source:
#   https://github.com/Klipper3d/klipper/blob/94604293e8b8baf919af261a61bd439df9d1a941/klippy/mcu.py#L282-L338

"""Bundled TriggerDispatch for Klipper forks that predate this class.

Ported from Klipper's ``klippy/mcu.py``.  Wraps chelper's ``trdispatch``
C functions and manages ``MCU_trsync`` instances for multi-MCU homing
synchronisation.

Both ``MCU_trsync`` and the chelper FFI functions exist on Klipper v0.12
(Qidi forks, etc.) — only the Python ``TriggerDispatch`` wrapper is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import chelper
from mcu import MCU_trsync

if TYPE_CHECKING:
    from cffi import FFI
    from mcu import MCU
    from reactor import ReactorCompletion
    from stepper import MCU_stepper

TRSYNC_TIMEOUT = 0.025
TRSYNC_SINGLE_MCU_TIMEOUT = 0.250


@final
class TriggerDispatch:
    """Multi-MCU trigger dispatch for homing synchronisation."""

    def __init__(self, mcu: MCU) -> None:
        self._mcu = mcu
        self._trigger_completion: ReactorCompletion | None = None
        ffi_main, ffi_lib = chelper.get_ffi()
        self._trdispatch: FFI.CData = ffi_main.gc(ffi_lib.trdispatch_alloc(), ffi_lib.free)
        self._trsyncs: list[MCU_trsync] = [MCU_trsync(mcu, self._trdispatch)]

    def get_oid(self) -> int:
        return self._trsyncs[0].get_oid()

    def get_command_queue(self) -> object:
        return self._trsyncs[0].get_command_queue()

    def add_stepper(self, stepper: MCU_stepper) -> None:
        trsyncs = {trsync.get_mcu(): trsync for trsync in self._trsyncs}
        trsync = trsyncs.get(stepper.get_mcu())
        if trsync is None:
            trsync = MCU_trsync(stepper.get_mcu(), self._trdispatch)
            self._trsyncs.append(trsync)
        trsync.add_stepper(stepper)
        # Check for unsupported multi-mcu shared stepper rails
        sname = stepper.get_name()
        if sname.startswith("stepper_"):
            for ot in self._trsyncs:
                for s in ot.get_steppers():
                    if ot is not trsync and s.get_name().startswith(sname[:9]):
                        cerror = self._mcu.get_printer().config_error
                        msg = "Multi-mcu homing not supported on multi-mcu shared axis"
                        raise cerror(msg)

    def get_steppers(self) -> list[MCU_stepper]:
        return [s for trsync in self._trsyncs for s in trsync.get_steppers()]

    def start(self, print_time: float) -> ReactorCompletion:
        reactor = self._mcu.get_printer().get_reactor()
        self._trigger_completion = reactor.completion()
        expire_timeout = TRSYNC_TIMEOUT
        if len(self._trsyncs) == 1:
            expire_timeout = TRSYNC_SINGLE_MCU_TIMEOUT
        for i, trsync in enumerate(self._trsyncs):
            report_offset = float(i) / len(self._trsyncs)
            trsync.start(print_time, report_offset, self._trigger_completion, expire_timeout)
        etrsync = self._trsyncs[0]
        _ffi_main, ffi_lib = chelper.get_ffi()
        ffi_lib.trdispatch_start(self._trdispatch, etrsync.REASON_HOST_REQUEST)
        return self._trigger_completion

    def wait_end(self, end_time: float) -> None:
        etrsync = self._trsyncs[0]
        etrsync.set_home_end_time(end_time)
        if self._mcu.is_fileoutput():
            assert self._trigger_completion is not None
            self._trigger_completion.complete(True)
        assert self._trigger_completion is not None
        _ = self._trigger_completion.wait()

    def stop(self) -> int:
        _ffi_main, ffi_lib = chelper.get_ffi()
        ffi_lib.trdispatch_stop(self._trdispatch)
        res = [trsync.stop() for trsync in self._trsyncs]
        # `==` not `>=`: v0.12 orders COMMS_TIMEOUT=2 below HOST_REQUEST=3 and PAST_END_TIME=4.
        err_res = [r for r in res if r == MCU_trsync.REASON_COMMS_TIMEOUT]
        if err_res:
            return err_res[0]
        return res[0]
