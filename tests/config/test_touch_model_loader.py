"""Focused tests: touch model loader inherits non-default global sampling config."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from cartographer.adapters.klipper.configuration import parse_touch_model_with_defaults
from cartographer.config.fields import parse
from cartographer.interfaces.configuration import TouchConfig

if TYPE_CHECKING:
    from configfile import ConfigWrapper


class _FakeSection:
    """Minimal structural stub for the Klipper ConfigWrapper section interface."""

    def __init__(self, name: str, values: dict[str, object]) -> None:
        self._name = name
        self._values = values

    def get_name(self) -> str:
        return self._name

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)

    def getint(self, key: str, default: object = None, minval: int | None = None, maxval: int | None = None) -> object:
        val = self._values.get(key, default)
        return int(val) if val is not None else default

    def getfloat(
        self,
        key: str,
        default: object = None,
        minval: float | None = None,
        maxval: float | None = None,
        note_valid: bool = True,
    ) -> object:
        val = self._values.get(key, default)
        return float(val) if val is not None else default

    def getboolean(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)

    def getfloatlist(self, key: str, count: int | None = None, default: object = None) -> object:
        return self._values.get(key, default)

    def getintlist(self, key: str, count: int | None = None, default: object = None) -> object:
        return self._values.get(key, default)

    def getchoice(self, key: str, choices: dict[str, str], default: object = None) -> object:
        val = self._values.get(key, default)
        return str(val) if val is not None else default


def _global_touch(overrides: dict[str, object] | None = None) -> TouchConfig:
    """Parse a TouchConfig from a fake section with optional overrides."""
    base: dict[str, object] = {
        "samples": 3,
        "max_samples": 10,
        "max_noisy_samples": 2,
        "UNSAFE_max_touch_temperature": 150,
        "EXPERIMENTAL_home_random_radius": 0.0,
        "retract_distance": 2.0,
        "sample_range": 0.010,
    }
    if overrides:
        base.update(overrides)
    section = cast("ConfigWrapper", _FakeSection("cartographer touch", base))
    return parse(TouchConfig, section, models={})


def _model_section(name: str, overrides: dict[str, object] | None = None) -> ConfigWrapper:
    """Build a fake touch_model section."""
    base: dict[str, object] = {
        "speed": 3.0,
        "z_offset": -0.05,
        "threshold": 1000,
    }
    if overrides:
        base.update(overrides)
    return cast("ConfigWrapper", _FakeSection(f"cartographer touch_model {name}", base))


def test_missing_fields_inherit_non_default_globals() -> None:
    """Model section without samples/sample_range picks up non-default global values.

    Legacy model sections written before these keys existed must end up with
    concrete values equal to the current global config, not the
    hard-coded option defaults.
    """
    global_touch = _global_touch({"samples": 7, "sample_range": 0.008})
    result = parse_touch_model_with_defaults(
        _model_section("legacy"),
        global_samples=global_touch.samples,
        global_sample_range=global_touch.sample_range,
    )
    assert result.samples == 7
    assert abs(result.sample_range - 0.008) < 1e-9


def test_explicit_model_fields_override_globals() -> None:
    """Explicit samples/sample_range in a model section win over the global config."""
    global_touch = _global_touch({"samples": 7, "sample_range": 0.008})
    result = parse_touch_model_with_defaults(
        _model_section("override", {"samples": 5, "sample_range": 0.012}),
        global_samples=global_touch.samples,
        global_sample_range=global_touch.sample_range,
    )
    assert result.samples == 5
    assert abs(result.sample_range - 0.012) < 1e-9
