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

    from cartographer.interfaces.printer import MacroParams

T = TypeVar("T")

_PARAM_METADATA_KEY = "_macro_param"


@dataclass(frozen=True)
class _ParamMeta:
    """Internal metadata attached to dataclass fields via field(metadata=...)."""

    description: str | None
    default: Any  # MISSING means required
    min: float | None
    max: float | None
    key: str | None
    parse_fn: Callable[[MacroParams], Any] | None

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
def param(
    description: str | None = ...,
    *,
    default: T,
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
    parse_fn: Callable[[MacroParams], T] | None = ...,
) -> T: ...


@overload
def param(
    description: str | None = ...,
    *,
    parse_fn: Callable[[MacroParams], T],
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
) -> T: ...


@overload
def param(
    description: str | None = ...,
    *,
    key: str | None = ...,
    min: float | None = ...,
    max: float | None = ...,
) -> Any: ...


def param(
    description: str | None = None,
    *,
    default: Any = MISSING,
    key: str | None = None,
    min: float | None = None,
    max: float | None = None,
    parse_fn: Callable[[MacroParams], Any] | None = None,
) -> Any:
    """Define a macro parameter on a frozen dataclass field.

    Fields with a description are documented; fields without are internal.

    Args:
        description: Human-readable description. Omit for internal fields.
        default: Default value. Omit for required fields. Type-checked against the field type.
        key: Macro parameter name override. Defaults to field name uppercased.
        min: Minimum value constraint (for numeric types, passed as minval).
        max: Maximum value constraint (for numeric types, passed as maxval).
        parse_fn: Custom parse function. Receives MacroParams, returns the field value.
                  When provided, type-based auto-parsing is skipped.
    """
    meta = _ParamMeta(
        description=description,
        default=default,
        min=min,
        max=max,
        key=key,
        parse_fn=parse_fn,
    )

    field_kwargs: dict[str, Any] = {
        "metadata": {_PARAM_METADATA_KEY: meta},
    }

    if default is not MISSING:
        field_kwargs["default"] = default

    return field(**field_kwargs)


def _get_param_meta(f: dataclasses.Field[Any]) -> _ParamMeta | None:
    """Extract ParamMeta from a dataclass field, or None if not a param()."""
    return f.metadata.get(_PARAM_METADATA_KEY)


# ---------------------------------------------------------------------------
# Macro-specific backend adapter
# ---------------------------------------------------------------------------


_TRUTHY = {"1", "yes", "true"}
_FALSY = {"0", "no", "false"}


class _MacroBackend:
    """Wraps MacroParams to satisfy the ParseBackend protocol."""

    def __init__(self, params: MacroParams) -> None:
        self._params: MacroParams = params

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
        return self._params.get_float(name, **kwargs)

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
        return self._params.get_int(name, **kwargs)

    def get_str(self, name: str, *, default: Any = MISSING) -> Any:
        kwargs: dict[str, Any] = {}
        if default is not MISSING:
            kwargs["default"] = default
        return self._params.get(name, **kwargs)

    def get_bool(self, name: str, *, default: Any = MISSING) -> Any:
        if default is MISSING:
            raw = self._params.get(name, default=None)
            if raw is None:
                msg = f"Error on '{name}': must provide a value (1/0, yes/no, true/false)"
                raise RuntimeError(msg)
        else:
            raw = self._params.get(name, default=None)
            if raw is None:
                return default

        lower = str(raw).lower()
        if lower in _TRUTHY:
            return True
        if lower in _FALSY:
            return False

        msg = f"Invalid boolean value '{raw}' for parameter '{name}'. Use 1/0, yes/no, or true/false."
        raise RuntimeError(msg)

    def get_enum(self, name: str, enum_type: type[Enum], *, default: Any = MISSING) -> Any:
        # Case-insensitive enum matching
        lower_mapping = {str(v.value).lower(): v for v in enum_type}

        if default is MISSING:
            raw = self._params.get(name)
        elif default is None:
            raw = self._params.get(name, default=None)
            if raw is None:
                return None
        else:
            raw = self._params.get(name, default=str(default.value))

        lower_choice = str(raw).lower()
        if lower_choice not in lower_mapping:
            valid = ", ".join(f"'{v.value}'" for v in enum_type)
            msg = f"Invalid choice '{raw}' for parameter '{name}'. Valid choices are: {valid}"
            raise RuntimeError(msg)

        return lower_mapping[lower_choice]


# ---------------------------------------------------------------------------
# Unknown-param validation
# ---------------------------------------------------------------------------


def _validate_unknown_params(cls: type, params: MacroParams) -> None:
    """Raise RuntimeError if the user passed any unknown macro parameters."""
    known: set[str] = set()
    for f in fields(cls):
        meta = _get_param_meta(f)
        if meta is not None:
            key = meta.key if meta.key is not None else f.name.upper()
            known.add(key.upper())

    # get_command_parameters() returns {"PARAM": "value", ...} for extended gcode commands.
    # Keys are already uppercased by Klipper.
    supplied = params.get_command_parameters()
    unknown_params = [k for k in supplied if k not in known]

    if unknown_params:
        valid = ", ".join(sorted(known)) if known else "(none)"
        unknown_list = ", ".join(sorted(unknown_params))
        msg = f"Unknown parameter(s): {unknown_list}. Valid parameters are: {valid}"
        raise RuntimeError(msg)


def validate_unknown_params(cls: type, params: MacroParams) -> None:
    """Raise RuntimeError if the user passed any unknown macro parameters.

    Public version of the validation used internally by parse().
    Useful when parsing is handled separately (e.g. complex classmethod factories)
    but you still want unknown-param validation.
    """
    _validate_unknown_params(cls, params)


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------


def parse(cls: type[T], params: MacroParams, **defaults: Any) -> T:
    """Parse a frozen dataclass from Klipper macro parameters.

    Fields with ``param()`` metadata are parsed automatically based on their type.
    Fields with ``parse_fn`` use the custom function.
    Fields without ``param()`` metadata must be provided via ``defaults``.

    Defaults provide config-derived fallback values: the macro parameter still takes
    priority if the user supplies it, but the default value is used when the
    parameter is absent.

    Unknown parameters (not declared as param() fields) raise RuntimeError.

    Args:
        cls: The dataclass type to parse into.
        params: Klipper macro parameters.
        **defaults: Config-derived defaults for param() fields. The macro parameter
            takes priority when supplied by the user.

    Returns:
        An instance of ``cls`` populated from macro params.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__name__} is not a dataclass"
        raise TypeError(msg)

    _validate_unknown_params(cls, params)

    # Get type hints — with from __future__ annotations, these are strings
    hints = {f.name: f.type for f in fields(cls)}

    kwargs: dict[str, Any] = {}
    backend = _MacroBackend(params)

    for f in fields(cls):
        meta = _get_param_meta(f)
        if meta is None:
            if f.name not in defaults:
                msg = f"Field '{f.name}' on {cls.__name__} has no param() metadata and no default provided"
                raise TypeError(msg)
            kwargs[f.name] = defaults[f.name]
            continue

        if meta.parse_fn is not None:
            kwargs[f.name] = meta.parse_fn(params)
            continue

        param_key = meta.key if meta.key is not None else f.name.upper()
        # Use default as the fallback if provided (config-derived defaults),
        # but still parse from macro params so user input takes priority.
        effective_meta = meta.field_meta
        if f.name in defaults:
            effective_meta = dataclasses.replace(effective_meta, default=defaults[f.name])

        type_hint = hints[f.name]
        kwargs[f.name] = parse_value(backend, param_key, type_hint, effective_meta, cls.__module__)

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Introspection (ParamInfo / get_all_params)
# ---------------------------------------------------------------------------

ParamInfo = FieldInfo


def get_all_params(cls: type) -> list[ParamInfo]:
    """Extract all documented param() fields from a dataclass as ParamInfo objects.

    Fields without a description (description is None) are excluded.

    Args:
        cls: The dataclass type to inspect.

    Returns:
        A list of ParamInfo for each documented param() field.
    """
    return get_all_fields(cls, _PARAM_METADATA_KEY, key_fn=lambda name, key: key if key is not None else name.upper())
