# pyright: reportExplicitAny=false, reportUnknownMemberType=false
# Any is unavoidable in this module due to dataclass field metadata and dynamic type resolution.
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    overload,
)

from cartographer.lib.fields import MISSING, FieldInfo, FieldMeta, get_all_fields, parse_value

if TYPE_CHECKING:
    from enum import Enum

    from configfile import ConfigWrapper

T = TypeVar("T")

_OPTION_METADATA_KEY = "_config_option"


@dataclass(frozen=True)
class _OptionMeta:
    """Internal metadata attached to dataclass fields via field(metadata=...)."""

    description: str | None
    default: Any  # MISSING means required
    min: float | None
    max: float | None
    key: str | None
    parse_fn: Callable[[ConfigWrapper], Any] | None

    @property
    def field_meta(self) -> FieldMeta:
        """Convert to shared FieldMeta for the generic parse/introspection layer."""
        return FieldMeta(
            description=self.description,
            default=self.default,
            min=self.min,
            max=self.max,
            key=self.key,
        )


@overload
def option(
    description: str | None = ...,
    *,
    default: T,
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
    parse_fn: Callable[[ConfigWrapper], T] | None = ...,
) -> T: ...


@overload
def option(
    description: str | None = ...,
    *,
    parse_fn: Callable[[ConfigWrapper], T],
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
) -> T: ...


@overload
def option(
    description: str | None = ...,
    *,
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
) -> Any: ...


def option(
    description: str | None = None,
    *,
    default: Any = MISSING,
    key: str | None = None,
    min: float | None = None,
    max: float | None = None,
    parse_fn: Callable[[ConfigWrapper], Any] | None = None,
) -> Any:
    """Define a config option on a frozen dataclass field.

    Fields with a description are documented; fields without are internal.

    Args:
        description: Human-readable description. Omit for internal fields.
        default: Default value. Omit for required fields. Type-checked against the field type.
        key: Config option name if different from field name (e.g. 'UNSAFE_max_touch_temperature').
        min: Minimum value constraint (for numeric types).
        max: Maximum value constraint (for numeric types).
        parse_fn: Custom parse function. Receives ConfigWrapper, returns the field value.
                  When provided, type-based auto-parsing is skipped.
    """
    meta = _OptionMeta(
        description=description,
        default=default,
        min=min,
        max=max,
        key=key,
        parse_fn=parse_fn,
    )

    field_kwargs: dict[str, Any] = {
        "metadata": {_OPTION_METADATA_KEY: meta},
    }

    if default is not MISSING:
        field_kwargs["default"] = default

    return field(**field_kwargs)


def _get_option_meta(f: dataclasses.Field[Any]) -> _OptionMeta | None:
    """Extract OptionMeta from a dataclass field, or None if not an option()."""
    return f.metadata.get(_OPTION_METADATA_KEY)


# ---------------------------------------------------------------------------
# Config-specific backend adapter
# ---------------------------------------------------------------------------


class _ConfigBackend:
    """Wraps a Klipper ConfigWrapper to satisfy the ParseBackend protocol."""

    def __init__(self, config: ConfigWrapper) -> None:
        self._config: ConfigWrapper = config

    def get_float(
        self,
        name: str,
        *,
        default: Any = MISSING,
        minval: float | None = None,
        maxval: float | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = default
        if minval is not None:
            kwargs["minval"] = minval
        if maxval is not None:
            kwargs["maxval"] = maxval
        return self._config.getfloat(name, **kwargs)

    def get_int(
        self,
        name: str,
        *,
        default: Any = MISSING,
        minval: int | None = None,
        maxval: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = default
        if minval is not None:
            kwargs["minval"] = minval
        if maxval is not None:
            kwargs["maxval"] = maxval
        return self._config.getint(name, **kwargs)

    def get_str(self, name: str, *, default: Any = MISSING) -> Any:
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = default
        return self._config.get(name, **kwargs)

    def get_bool(self, name: str, *, default: Any = MISSING) -> Any:
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = default
        return self._config.getboolean(name, **kwargs)

    def get_enum(self, name: str, enum_type: type[Enum], *, default: Any = MISSING) -> Any:
        choices = {str(m.value): str(m.value) for m in enum_type}
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = str(default.value)
        value = self._config.getchoice(name, choices, **kwargs)
        return enum_type(value)


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------


def parse(cls: type[T], config: ConfigWrapper, **overrides: Any) -> T:
    """Parse a frozen dataclass from a Klipper ConfigWrapper.

    Fields with ``option()`` metadata are parsed automatically based on their type.
    Fields with ``parse_fn`` use the custom function.
    Fields without ``option()`` metadata must be provided via ``overrides``.

    Args:
        cls: The dataclass type to parse into.
        config: Klipper ConfigWrapper for the relevant config section.
        **overrides: Values for fields that are not ``option()`` fields (e.g. models).

    Returns:
        An instance of ``cls`` populated from config.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__name__} is not a dataclass"
        raise TypeError(msg)

    # Get type hints — with from __future__ annotations, these are strings
    hints = {f.name: f.type for f in fields(cls)}

    kwargs: dict[str, Any] = {}
    backend = _ConfigBackend(config)

    for f in fields(cls):
        if f.name in overrides:
            kwargs[f.name] = overrides[f.name]
            continue

        meta = _get_option_meta(f)
        if meta is None:
            if f.name not in overrides:
                msg = f"Field '{f.name}' on {cls.__name__} has no option() metadata and no override provided"
                raise TypeError(msg)
            continue

        if meta.parse_fn is not None:
            kwargs[f.name] = meta.parse_fn(config)
            continue

        type_hint = hints[f.name]
        config_key = meta.key if meta.key is not None else f.name
        kwargs[f.name] = parse_value(backend, config_key, type_hint, meta.field_meta, cls.__module__)

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# get_option_name()
# ---------------------------------------------------------------------------


def get_option_name(cls: type, field_name: str) -> str:
    """Get the config option name for a dataclass field.

    Validates that the field exists and is an ``option()`` field.
    Returns the option name (``key`` if set, otherwise the field name).

    Args:
        cls: The dataclass type.
        field_name: The field name to look up.

    Returns:
        The config option name string.

    Raises:
        ValueError: If the field doesn't exist or isn't an option() field.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__name__} is not a dataclass"
        raise TypeError(msg)

    for f in fields(cls):
        if f.name == field_name:
            meta = _get_option_meta(f)
            if meta is None:
                msg = f"Field '{field_name}' on {cls.__name__} is not an option() field"
                raise ValueError(msg)
            return meta.key if meta.key is not None else f.name

    msg = f"Field '{field_name}' not found on {cls.__name__}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Introspection (OptionInfo / get_all_options)
# ---------------------------------------------------------------------------

OptionInfo = FieldInfo


def get_all_options(cls: type) -> list[OptionInfo]:
    """Extract all documented option() fields from a dataclass as OptionInfo objects.

    Fields without a description (description is None) are excluded.

    Args:
        cls: The dataclass type to inspect.

    Returns:
        A list of OptionInfo for each documented option() field.
    """
    return get_all_fields(cls, _OPTION_METADATA_KEY, key_fn=lambda name, key: key if key is not None else name)
