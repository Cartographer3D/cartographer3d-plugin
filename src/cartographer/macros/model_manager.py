from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.printer import Macro, MacroParams
from cartographer.macros.fields import param, parse

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelManagerParams:
    """Parameters for model manager macros."""

    load: str | None = param("Model name to load", default=None)
    remove: str | None = param("Model name to remove", default=None)


@final
class TouchModelManager(Macro):
    description: str = "Manage saved touch models"

    def __init__(self, mode: TouchMode, config: Configuration) -> None:
        self._mode = mode
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(ModelManagerParams, params)
        if p.load is not None:
            load = p.load.lower()
            logger.info("Loading touch model: %s", load)
            self._mode.load_model(load)
            return

        if p.remove is not None:
            remove = p.remove.lower()
            logger.info("Removing touch model: %s", remove)
            self._config.remove_touch_model(remove)
            return

        logger.info("Available touch models: %s", ", ".join(self._config.touch.models.keys()))


@final
class ScanModelManager(Macro):
    description: str = "Manage saved scan models"

    def __init__(self, mode: ScanMode, config: Configuration) -> None:
        self._mode = mode
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(ModelManagerParams, params)
        if p.load is not None:
            load = p.load.lower()
            logger.info("Loading scan model: %s", load)
            self._mode.load_model(load)
            return

        if p.remove is not None:
            remove = p.remove.lower()
            logger.info("Removing scan model: %s", remove)
            self._config.remove_scan_model(remove)
            return

        logger.info("Available scan models: %s", ", ".join(self._config.scan.models.keys()))
