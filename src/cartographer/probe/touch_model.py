from __future__ import annotations

from typing import final

from typing_extensions import override

from cartographer.interfaces.configuration import TouchModelConfiguration
from cartographer.probe.model import ModelSelectorMixin


@final
class TouchModel:
    def __init__(self, config: TouchModelConfiguration) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def z_offset(self) -> float:
        return self.config.z_offset

    @property
    def speed(self) -> float:
        return self.config.speed

    @property
    def threshold(self) -> int:
        return self.config.threshold

    @property
    def samples(self) -> int | None:
        return self.config.samples

    @property
    def sample_range(self) -> float | None:
        return self.config.sample_range


class TouchModelSelectorMixin(ModelSelectorMixin[TouchModel, TouchModelConfiguration]):
    @override
    def _create_model(self, config: TouchModelConfiguration) -> TouchModel:
        return TouchModel(config)
