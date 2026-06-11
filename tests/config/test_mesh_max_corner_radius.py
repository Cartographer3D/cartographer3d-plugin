from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from cartographer.config.fields import parse
from cartographer.interfaces.configuration import ScanConfig
from tests.config.test_fields import MockConfigWrapper

if TYPE_CHECKING:
    from configfile import ConfigWrapper


def parse_scan_config(value: str | None) -> ScanConfig:
    data = {} if value is None else {"mesh_max_corner_radius": value}
    config = MockConfigWrapper(data)
    return parse(ScanConfig, cast("ConfigWrapper", cast("object", config)), models={})


@pytest.mark.parametrize("value", [None, "auto", "AUTO", " Auto "])
def test_mesh_max_corner_radius_auto_values_parse_to_none(value: str | None) -> None:
    assert parse_scan_config(value).mesh_max_corner_radius is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("2", 2.0),
        ("2.5", 2.5),
    ],
)
def test_mesh_max_corner_radius_numeric_values_parse_to_float(value: str, expected: float) -> None:
    assert parse_scan_config(value).mesh_max_corner_radius == expected


@pytest.mark.parametrize("value", ["-1", "nope"])
def test_mesh_max_corner_radius_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        _ = parse_scan_config(value)
