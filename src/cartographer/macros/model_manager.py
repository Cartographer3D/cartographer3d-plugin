import logging
from typing import final

from typing_extensions import override

from cartographer.interfaces.configuration import Configuration
from cartographer.interfaces.printer import Macro, MacroParams
from cartographer.probe.scan_mode import ScanMode
from cartographer.probe.touch_mode import TouchMode

logger = logging.getLogger(__name__)


@final
class TouchModelManager(Macro):
    description: str = "Manage saved touch models"

    def __init__(self, mode: TouchMode, config: Configuration) -> None:
        self._mode = mode
        self._config = config

    @override
    def run(self, params: MacroParams) -> None:
        load = params.get("LOAD", None)
        if load is not None:
            load = load.lower()
            logger.info("Loading touch model: %s", load)
            self._mode.load_model(load)
            return

        remove = params.get("REMOVE", None)
        if remove is not None:
            remove = remove.lower()
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
        load = params.get("LOAD", None)
        if load is not None:
            load = load.lower()
            logger.info("Loading scan model: %s", load)
            self._mode.load_model(load)
            return

        remove = params.get("REMOVE", None)
        if remove is not None:
            remove = remove.lower()
            logger.info("Removing scan model: %s", remove)
            self._config.remove_scan_model(remove)
            return

        logger.info("Available scan models: %s", ", ".join(self._config.scan.models.keys()))
