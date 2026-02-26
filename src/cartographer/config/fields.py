# pyright: reportExplicitAny=false, reportUnknownMemberType=false
# Any is unavoidable in this module due to dataclass field metadata and dynamic type resolution.
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from configfile import ConfigWrapper

T = TypeVar("T")

_MISSING = object()
_OPTION_METADATA_KEY = "_config_option"


@dataclass(frozen=True)
class _OptionMeta:
    """Internal metadata attached to dataclass fields via field(metadata=...)."""

    description: str | None
    default: Any  # _MISSING means required
    min: float | None
    max: float | None
    key: str | None
    parse_fn: Callable[[ConfigWrapper], Any] | None


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
    default: Any = _MISSING,
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

    if default is not _MISSING:
        field_kwargs["default"] = default

    return field(**field_kwargs)


def _get_option_meta(f: dataclasses.Field[Any]) -> _OptionMeta | None:
    """Extract OptionMeta from a dataclass field, or None if not an option()."""
    return f.metadata.get(_OPTION_METADATA_KEY)


def _resolve_type(type_hint: Any, module_name: str | None = None) -> type:
    """Resolve a type hint string (from __future__ annotations) to a real type.

    Handles basic types used in config: float, int, str, bool, Enum subclasses,
    and Optional variants.
    """
    if isinstance(type_hint, str):
        type_map: dict[str, type] = {
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
        }

        base = type_hint.replace(" ", "").split("|")[0]
        resolved = type_map.get(base)
        if resolved is not None:
            return resolved

        # Try resolving from the dataclass's module (for Enum subclasses, etc.)
        if module_name is not None:
            module = sys.modules.get(module_name)
            if module is not None:
                resolved = getattr(module, base, None)
                if isinstance(resolved, type):
                    return resolved

        msg = f"Cannot resolve type hint '{type_hint}' for config parsing"
        raise TypeError(msg)

    if isinstance(type_hint, type):
        return type_hint

    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        return origin

    msg = f"Cannot resolve type hint '{type_hint}' for config parsing"
    raise TypeError(msg)


def _is_optional(type_hint: Any) -> bool:
    """Check if a type hint is Optional (X | None)."""
    if isinstance(type_hint, str):
        return "None" in type_hint or "| None" in type_hint
    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        import typing

        if origin is typing.Union:
            args = getattr(type_hint, "__args__", ())
            return type(None) in args
    return False


def _parse_field_value(
    config: ConfigWrapper,
    name: str,
    type_hint: Any,
    meta: _OptionMeta,
    module_name: str | None,
) -> Any:
    """Parse a single field value from a ConfigWrapper based on type and metadata."""
    is_required = meta.default is _MISSING
    is_nullable = _is_optional(type_hint)
    resolved_type = _resolve_type(type_hint, module_name)

    if resolved_type is float:
        kwargs: dict[str, Any] = {}
        if meta.min is not None:
            kwargs["minval"] = meta.min
        if meta.max is not None:
            kwargs["maxval"] = meta.max

        if is_required:
            return config.getfloat(name, **kwargs)

        if is_nullable and meta.default is None:
            return config.getfloat(name, default=None, **kwargs)

        return config.getfloat(name, default=meta.default, **kwargs)

    if resolved_type is int:
        kwargs = {}
        if meta.min is not None:
            kwargs["minval"] = int(meta.min)
        if meta.max is not None:
            kwargs["maxval"] = int(meta.max)

        if is_required:
            return config.getint(name, **kwargs)

        if is_nullable and meta.default is None:
            return config.getint(name, default=None, **kwargs)

        return config.getint(name, default=meta.default, **kwargs)

    if resolved_type is bool:
        if is_required:
            return config.getboolean(name)

        if is_nullable and meta.default is None:
            return config.getboolean(name, default=None)

        return config.getboolean(name, default=meta.default)

    if resolved_type is str:
        if is_nullable and (meta.default is None or meta.default is _MISSING):
            return config.get(name, default=None)

        if is_required:
            return config.get(name)

        return config.get(name, default=meta.default)

    if issubclass(resolved_type, Enum):
        choices = {str(m.value): str(m.value) for m in resolved_type}
        if is_required:
            value = config.getchoice(name, choices)
        else:
            value = config.getchoice(name, choices, default=str(meta.default.value))
        return resolved_type(value)

    msg = f"Unsupported type '{type_hint}' for auto-parsing field '{name}'. Use parse_fn instead."
    raise TypeError(msg)


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
        kwargs[f.name] = _parse_field_value(config, config_key, type_hint, meta, cls.__module__)

    return cls(**kwargs)


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
