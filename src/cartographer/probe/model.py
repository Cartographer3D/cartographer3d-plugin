from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from cartographer.lib.str import removesuffix

TModel = TypeVar("TModel")
TModelConfig = TypeVar("TModelConfig")

logger = logging.getLogger(__name__)


class ModelSelectorMixin(Generic[TModel, TModelConfig], ABC):
    def __init__(self, models: dict[str, TModelConfig]) -> None:
        self._models: dict[str, TModelConfig] = models
        self._loaded_model: TModel | None = None

    def get_model(self) -> TModel:
        if self._loaded_model is None:
            msg = f"{self._get_type()} model not loaded."
            raise RuntimeError(msg)
        return self._loaded_model

    def has_model(self) -> bool:
        return self._loaded_model is not None

    def load_model(self, name: str):
        model = self._models.get(name, None)
        if model is None:
            msg = f"{self._get_type()} model {name} not found"
            raise RuntimeError(msg)
        self._loaded_model = self._create_model(self._models[name])

    def _get_type(self) -> str:
        """Dynamically retrieve the class name for type."""
        return removesuffix(self.__class__.__name__, "Mode")

    @abstractmethod
    def _create_model(self, config: TModelConfig) -> TModel:
        """Override in subclass to build model from config."""
        raise NotImplementedError
