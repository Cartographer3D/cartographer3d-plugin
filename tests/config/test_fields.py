from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from enum import Enum
from typing import final

import pytest

from cartographer.config.fields import get_option_name, option, parse
from cartographer.interfaces.configuration import MeshDirection, MeshPath

# ---------------------------------------------------------------------------
# Mock ConfigWrapper — mirrors Klipper's configfile.ConfigWrapper behaviour
# ---------------------------------------------------------------------------

_sentinel: object = object()


@final
class MockConfigWrapper:
    """Minimal mock of Klipper's ConfigWrapper for unit-testing parse().

    Pass a dict[str, str] of option→value pairs (as they'd appear in a
    Klipper config file).  The getXxx helpers mimic Klipper's real
    parsing: look up in the dict, fall back to default, raise on missing
    required options.
    """

    error: type[configparser.Error] = configparser.Error

    def __init__(self, data: dict[str, str] | None = None, *, section: str = "test") -> None:
        self._data: dict[str, str] = data or {}
        self.section: str = section

    def get_name(self) -> str:
        return self.section

    # --- str ---

    def get(self, option: str, default: str | None = _sentinel, **_kw: object) -> str | None:  # pyright: ignore[reportArgumentType]
        value = self._data.get(option)
        if value is not None:
            return value
        if default is not _sentinel:
            return default
        msg = f"Option '{option}' in section '{self.section}' must be specified"
        raise self.error(msg)

    # --- int ---

    def getint(self, option: str, default: int | None = _sentinel, **_kw: object) -> int | None:  # pyright: ignore[reportArgumentType]
        value = self._data.get(option)
        if value is not None:
            return int(value)
        if default is not _sentinel:
            return default
        msg = f"Option '{option}' in section '{self.section}' must be specified"
        raise self.error(msg)

    # --- float ---

    def getfloat(self, option: str, default: float | None = _sentinel, **_kw: object) -> float | None:  # pyright: ignore[reportArgumentType]
        value = self._data.get(option)
        if value is not None:
            return float(value)
        if default is not _sentinel:
            return default
        msg = f"Option '{option}' in section '{self.section}' must be specified"
        raise self.error(msg)

    # --- bool ---

    def getboolean(self, option: str, default: bool | None = _sentinel, **_kw: object) -> bool | None:  # pyright: ignore[reportArgumentType]
        value = self._data.get(option)
        if value is not None:
            return value.lower() in ("true", "yes", "1")
        if default is not _sentinel:
            return default
        msg = f"Option '{option}' in section '{self.section}' must be specified"
        raise self.error(msg)

    # --- choice ---

    def getchoice(self, option: str, choices: dict[str, str], default: str = _sentinel, **_kw: object) -> str:  # pyright: ignore[reportArgumentType]
        value = self._data.get(option)
        if value is None:
            if default is _sentinel:
                msg = f"Option '{option}' in section '{self.section}' must be specified"
                raise self.error(msg)
            value = default
        if value not in choices:
            msg = f"Choice '{value}' for option '{option}' in section '{self.section}' is not a valid choice"
            raise self.error(msg)
        return choices[value]


# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


class Colour(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

    def __str__(self) -> str:  # noqa: PLE0307  # pyright: ignore[reportImplicitOverride]
        return self.value


@dataclass(frozen=True)
class SimpleConfig:
    name: str = option("A name")
    speed: float = option("Speed", default=100.0)
    count: int = option("Count", default=5)
    enabled: bool = option("Enabled", default=True)


@dataclass(frozen=True)
class EnumConfig:
    colour: Colour = option("A colour", default=Colour.RED)


@dataclass(frozen=True)
class RequiredFieldConfig:
    name: str = option("A required string")
    speed: float = option("A required float")


@dataclass(frozen=True)
class OptionalConfig:
    label: str | None = option("An optional string", default=None)
    weight: float | None = option("An optional float", default=None)


@dataclass(frozen=True)
class KeyRemapConfig:
    max_temp: float = option("Maximum temperature", default=150.0, key="UNSAFE_max_temperature")


@dataclass(frozen=True)
class ParseFnConfig:
    section_name: str = option(None, parse_fn=lambda c: c.get_name().split(" ")[-1])


@dataclass(frozen=True)
class ConstrainedConfig:
    speed: float = option("Speed", default=50.0, min=0.0, max=200.0)


@dataclass(frozen=True)
class OverrideConfig:
    name: str = option("A name", default="test")
    models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredEnumConfig:
    colour: Colour = option("A required colour")


# ---------------------------------------------------------------------------
# Tests: parse() — basic types with defaults
# ---------------------------------------------------------------------------


class TestParseDefaults:
    def test_uses_defaults_when_config_empty(self) -> None:
        config = MockConfigWrapper({"name": "hello"})
        result = parse(SimpleConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.name == "hello"
        assert result.speed == 100.0
        assert result.count == 5
        assert result.enabled is True

    def test_overrides_defaults_from_config(self) -> None:
        config = MockConfigWrapper({"name": "world", "speed": "42.5", "count": "3", "enabled": "false"})
        result = parse(SimpleConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.name == "world"
        assert result.speed == 42.5
        assert result.count == 3
        assert result.enabled is False


# ---------------------------------------------------------------------------
# Tests: parse() — required fields
# ---------------------------------------------------------------------------


class TestParseRequired:
    def test_parses_required_fields(self) -> None:
        config = MockConfigWrapper({"name": "test", "speed": "99.9"})
        result = parse(RequiredFieldConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.name == "test"
        assert result.speed == 99.9

    def test_raises_on_missing_required_field(self) -> None:
        config = MockConfigWrapper({})
        with pytest.raises(configparser.Error):
            _ = parse(RequiredFieldConfig, config)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Tests: parse() — optional (nullable) fields
# ---------------------------------------------------------------------------


class TestParseOptional:
    def test_defaults_to_none(self) -> None:
        config = MockConfigWrapper({})
        result = parse(OptionalConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.label is None
        assert result.weight is None

    def test_parses_value_when_present(self) -> None:
        config = MockConfigWrapper({"label": "hello", "weight": "1.5"})
        result = parse(OptionalConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.label == "hello"
        assert result.weight == 1.5


# ---------------------------------------------------------------------------
# Tests: parse() — enum fields
# ---------------------------------------------------------------------------


class TestParseEnum:
    def test_uses_enum_default(self) -> None:
        config = MockConfigWrapper({})
        result = parse(EnumConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.colour is Colour.RED

    def test_parses_enum_from_config(self) -> None:
        config = MockConfigWrapper({"colour": "green"})
        result = parse(EnumConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.colour is Colour.GREEN

    def test_raises_on_invalid_enum_value(self) -> None:
        config = MockConfigWrapper({"colour": "purple"})
        with pytest.raises(configparser.Error, match="not a valid choice"):
            _ = parse(EnumConfig, config)  # pyright: ignore[reportArgumentType]

    def test_required_enum_parses(self) -> None:
        config = MockConfigWrapper({"colour": "blue"})
        result = parse(RequiredEnumConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.colour is Colour.BLUE

    def test_required_enum_raises_when_missing(self) -> None:
        config = MockConfigWrapper({})
        with pytest.raises(configparser.Error):
            _ = parse(RequiredEnumConfig, config)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Tests: parse() — key remapping
# ---------------------------------------------------------------------------


class TestParseKeyRemap:
    def test_reads_from_remapped_key(self) -> None:
        config = MockConfigWrapper({"UNSAFE_max_temperature": "200"})
        result = parse(KeyRemapConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.max_temp == 200.0

    def test_uses_default_when_remapped_key_absent(self) -> None:
        config = MockConfigWrapper({})
        result = parse(KeyRemapConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.max_temp == 150.0


# ---------------------------------------------------------------------------
# Tests: parse() — parse_fn
# ---------------------------------------------------------------------------


class TestParseFn:
    def test_uses_parse_fn(self) -> None:
        config = MockConfigWrapper({}, section="cartographer my_model")
        result = parse(ParseFnConfig, config)  # pyright: ignore[reportArgumentType]
        assert result.section_name == "my_model"


# ---------------------------------------------------------------------------
# Tests: parse() — overrides (non-option fields)
# ---------------------------------------------------------------------------


class TestParseOverrides:
    def test_passes_overrides(self) -> None:
        config = MockConfigWrapper({})
        result = parse(OverrideConfig, config, models={"a": "b"})  # pyright: ignore[reportArgumentType]
        assert result.name == "test"
        assert result.models == {"a": "b"}

    def test_non_option_field_without_override_raises(self) -> None:
        config = MockConfigWrapper({})
        with pytest.raises(TypeError, match="no option.*metadata and no override"):
            _ = parse(OverrideConfig, config)  # pyright: ignore[reportArgumentType]

    def test_override_takes_precedence_over_config(self) -> None:
        config = MockConfigWrapper({"name": "from_config"})
        result = parse(OverrideConfig, config, name="from_override", models={})  # pyright: ignore[reportArgumentType]
        assert result.name == "from_override"


# ---------------------------------------------------------------------------
# Tests: get_option_name()
# ---------------------------------------------------------------------------


class TestGetOptionName:
    def test_returns_field_name_by_default(self) -> None:
        assert get_option_name(SimpleConfig, "speed") == "speed"

    def test_returns_remapped_key(self) -> None:
        assert get_option_name(KeyRemapConfig, "max_temp") == "UNSAFE_max_temperature"

    def test_raises_on_unknown_field(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _ = get_option_name(SimpleConfig, "nonexistent")

    def test_raises_on_non_option_field(self) -> None:
        with pytest.raises(ValueError, match="not an option"):
            _ = get_option_name(OverrideConfig, "models")


# ---------------------------------------------------------------------------
# Tests: enum __str__ safety
# ---------------------------------------------------------------------------


class TestEnumStr:
    """Verify that str(Enum) returns the value, not 'ClassName.MEMBER'.

    On Python 3.11+ the default str() for str,Enum changed to return
    'ClassName.MEMBER'. Our enums override __str__ as a safety net.
    """

    def test_str_returns_value(self) -> None:
        assert str(Colour.RED) == "red"
        assert str(Colour.GREEN) == "green"
        assert str(Colour.BLUE) == "blue"

    def test_fstring_returns_value(self) -> None:
        assert f"{Colour.RED}" == "red"

    def test_mesh_direction_str(self) -> None:
        assert str(MeshDirection.X) == "x"
        assert str(MeshDirection.Y) == "y"

    def test_mesh_path_str(self) -> None:
        assert str(MeshPath.SNAKE) == "snake"
        assert str(MeshPath.ALTERNATING_SNAKE) == "alternating_snake"
        assert str(MeshPath.SPIRAL) == "spiral"
        assert str(MeshPath.RANDOM) == "random"

    def test_mesh_path_fstring(self) -> None:
        assert f"{MeshPath.SNAKE}" == "snake"

    def test_lower_works_without_str_override(self) -> None:
        """str methods like .lower() operate on the value directly."""

        assert MeshDirection.X.lower() == "x"
        assert MeshPath.ALTERNATING_SNAKE.lower() == "alternating_snake"


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_parse_rejects_non_dataclass(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="is not a dataclass"):
            _ = parse(NotADataclass, MockConfigWrapper())  # pyright: ignore[reportArgumentType]

    def test_get_option_name_rejects_non_dataclass(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="is not a dataclass"):
            _ = get_option_name(NotADataclass, "foo")

    def test_unresolvable_type_raises(self) -> None:
        @dataclass(frozen=True)
        class BadTypeConfig:
            data: list[int] = option("A list")

        with pytest.raises(TypeError, match="Cannot resolve type hint"):
            _ = parse(BadTypeConfig, MockConfigWrapper({}))  # pyright: ignore[reportArgumentType]

    def test_non_option_field_without_override_raises(self) -> None:
        @dataclass(frozen=True)
        class NoDefaultConfig:
            extra: str = field()
            name: str = option("A name", default="ok")

        with pytest.raises(TypeError, match="no option.*metadata and no override"):
            _ = parse(NoDefaultConfig, MockConfigWrapper({}))  # pyright: ignore[reportArgumentType]
