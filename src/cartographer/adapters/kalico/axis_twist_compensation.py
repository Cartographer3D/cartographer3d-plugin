from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter


class KalicoAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        return self.compensation.get_z_compensation_value([x, y, 0])
