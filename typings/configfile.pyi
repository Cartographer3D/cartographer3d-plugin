# https://github.com/Klipper3d/klipper/blob/master/klippy/configfile.py
import configparser
from typing import overload

from klippy import Printer

error = configparser.Error

class ConfigWrapper:
    error: type[configparser.Error]
    printer: Printer
    fileconfig: configparser.ConfigParser
    access_tracking: dict[str, str]
    section: str

    def __init__(
        self, printer: Printer, fileconfig: configparser.ConfigParser, access_tracking: dict[str, str], section: str
    ) -> None: ...
    def get_printer(self) -> Printer: ...
    def get_name(self) -> str: ...
    def getsection(self, section: str) -> ConfigWrapper: ...
    def has_section(self, section: str) -> bool: ...
    def get_prefix_sections(self, prefix: str) -> list[ConfigWrapper]: ...
    def deprecate(self, option: str, value: str | None = None) -> None: ...
    @overload
    def get(
        self,
        option: str,
        default: str = ...,
        note_valid: bool = True,
    ) -> str: ...
    @overload
    def get(
        self,
        option: str,
        default: None,
        note_valid: bool = True,
    ) -> str | None: ...
    @overload
    def getint(
        self,
        option: str,
        default: int = ...,
        minval: int | None = None,
        maxval: int | None = None,
        note_valid: bool = True,
    ) -> int: ...
    @overload
    def getint(
        self,
        option: str,
        default: None,
        minval: int | None = None,
        maxval: int | None = None,
        note_valid: bool = True,
    ) -> int | None: ...
    @overload
    def getfloat(
        self,
        option: str,
        default: float = ...,
        minval: float | None = None,
        maxval: float | None = None,
        above: float | None = None,
        below: float | None = None,
        note_valid: bool = True,
    ) -> float: ...
    @overload
    def getfloat(
        self,
        option: str,
        default: None,
        minval: float | None = None,
        maxval: float | None = None,
        above: float | None = None,
        below: float | None = None,
        note_valid: bool = True,
    ) -> float | None: ...
    @overload
    def getboolean(
        self,
        option: str,
        default: bool = ...,
        note_valid: bool = True,
    ) -> bool: ...
    @overload
    def getboolean(
        self,
        option: str,
        default: None,
        note_valid: bool = True,
    ) -> bool | None: ...
    def getchoice(
        self,
        option: str,
        choices: dict[str, str],
        default: str = ...,
        note_valid: bool = True,
    ) -> str: ...
    @overload
    def getintlist(
        self,
        option: str,
        default: list[int] = ...,
        sep: str = ",",
        count: int | None = None,
        note_valid: bool = True,
    ) -> list[int]: ...
    @overload
    def getintlist(
        self,
        option: str,
        default: None,
        sep: str = ",",
        count: int | None = None,
        note_valid: bool = True,
    ) -> list[int] | None: ...
    @overload
    def getfloatlist(
        self,
        option: str,
        default: list[float] = ...,
        sep: str = ",",
        count: int | None = None,
        note_valid: bool = True,
    ) -> list[float]: ...
    @overload
    def getfloatlist(
        self,
        option: str,
        default: None,
        sep: str = ",",
        count: int | None = None,
        note_valid: bool = True,
    ) -> list[float] | None: ...

class PrinterConfig:
    def get_printer(self) -> Printer: ...
    def runtime_warning(self, msg: str) -> None: ...
    def deprecate(self, option: str, value: str | None = None) -> None: ...
    def set(self, section: str, option: str, value: object) -> None: ...
    def remove_section(self, section: str) -> None: ...
