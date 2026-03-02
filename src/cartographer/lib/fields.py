# pyright: reportExplicitAny=false, reportUnknownMemberType=false
# Any is unavoidable in this module due to dataclass field metadata and dynamic type resolution.
from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from enum import Enum
from typing import (
    Any,
    Callable,
    Protocol,
)

MISSING = object()


@dataclass(frozen=True)
class FieldMeta:
    """Shared metadata shape for declarative field descriptors (config options & macro params)."""

    description: str | None
    default: Any  # MISSING means required
    min: float | None
    max: float | None
    key: str | None


def _resolve_type(type_hint: Any, module_name: str | None = None) -> type:
    """Resolve a type hint string (from __future__ annotations) to a real type.

    Handles basic types used in field parsing: float, int, str, bool, Enum subclasses,
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

        msg = f"Cannot resolve type hint '{type_hint}' for field parsing"
        raise TypeError(msg)

    if isinstance(type_hint, type):
        return type_hint

    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        return origin

    msg = f"Cannot resolve type hint '{type_hint}' for field parsing"
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


class ParseBackend(Protocol):
    """Adapter interface for type-dispatched field parsing.

    Backends wrap a concrete source (ConfigWrapper, MacroParams) and expose
    a uniform API so the shared parse_value logic stays source-agnostic.
    """

    def get_float(
        self,
        name: str,
        *,
        default: Any = MISSING,
        minval: float | None = None,
        maxval: float | None = None,
    ) -> Any: ...

    def get_int(
        self,
        name: str,
        *,
        default: Any = MISSING,
        minval: int | None = None,
        maxval: int | None = None,
    ) -> Any: ...

    def get_str(self, name: str, *, default: Any = MISSING) -> Any: ...

    def get_bool(self, name: str, *, default: Any = MISSING) -> Any: ...

    def get_enum(self, name: str, enum_type: type[Enum], *, default: Any = MISSING) -> Any: ...


def parse_value(
    backend: ParseBackend,
    name: str,
    type_hint: Any,
    meta: FieldMeta,
    module_name: str | None,
) -> Any:
    """Parse a single field value via the given backend, dispatching on resolved type."""
    is_required = meta.default is MISSING
    is_nullable = _is_optional(type_hint)
    resolved_type = _resolve_type(type_hint, module_name)

    if resolved_type is float:
        if is_required:
            default: Any = MISSING
        elif is_nullable and meta.default is None:
            default = None
        else:
            default = meta.default
        return backend.get_float(name, default=default, minval=meta.min, maxval=meta.max)

    if resolved_type is int:
        if is_required:
            default = MISSING
        elif is_nullable and meta.default is None:
            default = None
        else:
            default = meta.default
        return backend.get_int(
            name,
            default=default,
            minval=int(meta.min) if meta.min is not None else None,
            maxval=int(meta.max) if meta.max is not None else None,
        )

    if resolved_type is bool:
        if is_required:
            default = MISSING
        elif is_nullable and meta.default is None:
            default = None
        else:
            default = meta.default
        return backend.get_bool(name, default=default)

    if resolved_type is str:
        if is_nullable and (meta.default is None or meta.default is MISSING):
            return backend.get_str(name, default=None)
        if is_required:
            return backend.get_str(name)
        return backend.get_str(name, default=meta.default)

    if issubclass(resolved_type, Enum):
        if is_required:
            default = MISSING
        elif meta.default is None:
            default = None
        else:
            default = meta.default
        return backend.get_enum(name, resolved_type, default=default)

    msg = f"Unsupported type '{type_hint}' for auto-parsing field '{name}'. Use parse_fn instead."
    raise TypeError(msg)


@dataclass(frozen=True)
class FieldInfo:
    """Public metadata for a single declared field, used for docs generation."""

    name: str
    type: str
    description: str | None
    default: Any
    required: bool
    min: float | None
    max: float | None
    choices: list[str] | None

    @property
    def has_default(self) -> bool:
        """Whether this field has a default value."""
        return self.default is not MISSING


def get_all_fields(
    cls: type,
    metadata_key: str,
    *,
    key_fn: Callable[[str, str | None], str],
) -> list[FieldInfo]:
    """Extract all documented fields from a dataclass as FieldInfo objects.

    Fields without a description (description is None) are excluded.

    Args:
        cls: The dataclass type to inspect.
        metadata_key: The metadata dict key that stores the FieldMeta.
        key_fn: Maps (field_name, meta.key) to the display name.

    Returns:
        A list of FieldInfo for each documented field.
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__name__} is not a dataclass"
        raise TypeError(msg)

    result: list[FieldInfo] = []
    for f in fields(cls):
        raw_meta = f.metadata.get(metadata_key)
        if raw_meta is None:
            continue

        # Extract the FieldMeta — the domain-specific meta must expose these attributes
        meta: FieldMeta = _extract_field_meta(raw_meta)
        if meta.description is None:
            continue

        display_key = key_fn(f.name, meta.key)
        type_hint = f.type
        is_required = meta.default is MISSING

        # Detect Enum choices
        choices: list[str] | None = None
        try:
            resolved = _resolve_type(type_hint, cls.__module__)
            if issubclass(resolved, Enum):
                choices = [str(m.value) for m in resolved]
        except TypeError:
            pass

        result.append(
            FieldInfo(
                name=display_key,
                type=type_hint if isinstance(type_hint, str) else type_hint.__name__,
                description=meta.description,
                default=meta.default if not is_required else MISSING,
                required=is_required,
                min=meta.min,
                max=meta.max,
                choices=choices,
            )
        )

    return result


def _extract_field_meta(raw_meta: Any) -> FieldMeta:
    """Extract a FieldMeta from domain-specific metadata.

    Supports both FieldMeta instances directly and domain-specific meta objects
    that have a `field_meta` attribute.
    """
    if isinstance(raw_meta, FieldMeta):
        return raw_meta
    # Domain-specific meta (e.g. _OptionMeta, _ParamMeta) must expose field_meta
    field_meta = getattr(raw_meta, "field_meta", None)
    if isinstance(field_meta, FieldMeta):
        return field_meta
    msg = f"Cannot extract FieldMeta from {type(raw_meta).__name__}"
    raise TypeError(msg)
