# https://github.com/Klipper3d/klipper/blob/master/klippy/kinematics/extruder.py
from extras.heaters import Heater

class Extruder:
    def get_name(self) -> str: ...
    def get_heater(self) -> Heater: ...
