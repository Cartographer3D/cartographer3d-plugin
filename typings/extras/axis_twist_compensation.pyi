class AxisTwistCompensation:
    horizontal_move_z: float
    speed: float

    calibrate_start_x: float | None
    calibrate_end_x: float | None
    calibrate_y: float | None
    z_compensations: list[float]
    compensation_start_x: float | None
    compensation_end_x: float | None

    calibrate_start_y: float | None
    calibrate_end_y: float | None
    calibrate_x: float | None
    zy_compensations: list[float]
    compensation_start_y: float | None
    compensation_end_y: float | None

    def clear_compensations(self, axis: str) -> None: ...
    def get_z_compensation_value(self, pos: list[float]) -> float:
        """Only available in Kalico"""
        ...
