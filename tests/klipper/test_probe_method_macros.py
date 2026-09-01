"""Unit tests for build_probe_method_macros helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cartographer.adapters.klipper_like.utils import build_probe_method_macros


def _make_printer_config(sections: set[str]) -> tuple[MagicMock, MagicMock]:
    config = MagicMock()

    def has_section(name: str) -> bool:
        return name in sections

    def getsection(_name: str) -> MagicMock:
        return MagicMock()

    config.has_section.side_effect = has_section
    config.getsection.side_effect = getsection
    printer = MagicMock()
    return printer, config


_CASES: list[tuple[set[str], tuple[str, ...] | None, list[str]]] = [
    (set(), None, ["BED_MESH_CALIBRATE"]),
    ({"z_tilt"}, None, ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST"]),
    (
        {"z_tilt", "quad_gantry_level", "screws_tilt_adjust"},
        None,
        ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST", "QUAD_GANTRY_LEVEL", "SCREWS_TILT_CALCULATE"],
    ),
    ({"z_tilt"}, ("z_tilt", "z_tilt_ng"), ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST"]),
    ({"z_tilt_ng"}, ("z_tilt", "z_tilt_ng"), ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST"]),
    ({"z_tilt", "z_tilt_ng"}, ("z_tilt", "z_tilt_ng"), ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST"]),
    (
        {"z_tilt_ng", "quad_gantry_level", "screws_tilt_adjust"},
        ("z_tilt", "z_tilt_ng"),
        ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST", "QUAD_GANTRY_LEVEL", "SCREWS_TILT_CALCULATE"],
    ),
]


@pytest.mark.parametrize(("sections", "z_tilt_sections", "expected"), _CASES)
def test_build_probe_method_macros(
    sections: set[str], z_tilt_sections: tuple[str, ...] | None, expected: list[str]
) -> None:
    printer, config = _make_printer_config(sections)
    result = (
        build_probe_method_macros(printer, config, z_tilt_sections=z_tilt_sections)
        if z_tilt_sections is not None
        else build_probe_method_macros(printer, config)
    )
    assert result == expected
