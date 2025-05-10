from typing import final

from typing_extensions import override

from cartographer.interfaces.configuration import TouchModelConfiguration
from cartographer.probe.model import ModelSelectorMixin


@final
class TouchModel:
    def __init__(self, config: TouchModelConfiguration) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def z_offset(self) -> float:
        return self._config.z_offset

    @property
    def speed(self) -> float:
        return self._config.speed

    @property
    def threshold(self) -> int:
        return self._config.threshold


class TouchModelSelectorMixin(ModelSelectorMixin[TouchModel, TouchModelConfiguration]):
    @override
    def _create_model(self, config: TouchModelConfiguration) -> TouchModel:
        return TouchModel(config)
