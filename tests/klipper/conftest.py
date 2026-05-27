import sys
from unittest.mock import Mock

# Stub Klipper modules not present in the test environment.
_mcu_mock = Mock()
_mcu_mock.MCU_endstop = object  # Real base class so inheritance works
_extras_mock = Mock()
_extras_mock.manual_probe = Mock(spec=[])  # Empty spec so hasattr(manual_probe, "ProbeResult") is False
for _name, _mock in (
    ("mcu", _mcu_mock),
    ("gcode", Mock()),
    ("pins", Mock()),
    ("extras", _extras_mock),
    ("extras.manual_probe", _extras_mock.manual_probe),
):
    sys.modules[_name] = _mock
