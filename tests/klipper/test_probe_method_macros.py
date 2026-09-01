"""Unit tests for build_probe_method_macros helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cartographer.adapters.klipper_like.utils import build_probe_method_macros


def _make_printer_config(sections: set[str]) -> tuple[MagicMock, MagicMock]:
    """Return (printer, config) mocks where has_section reflects *sections*."""
    config = MagicMock()

    def has_section(name: str) -> bool:
        return name in sections

    def getsection(_name: str) -> MagicMock:
        return MagicMock()

    config.has_section.side_effect = has_section
    config.getsection.side_effect = getsection
    printer = MagicMock()
    return printer, config


class TestDefaultZTilt:
    def test_empty_config_returns_only_bed_mesh(self) -> None:
        printer, config = _make_printer_config(set())
        assert build_probe_method_macros(printer, config) == ["BED_MESH_CALIBRATE"]

    def test_z_tilt_adds_z_tilt_adjust(self) -> None:
        printer, config = _make_printer_config({"z_tilt"})
        result = build_probe_method_macros(printer, config)
        assert result == ["BED_MESH_CALIBRATE", "Z_TILT_ADJUST"]

    def test_full_config_returns_ordered_list(self) -> None:
        printer, config = _make_printer_config({"z_tilt", "quad_gantry_level", "screws_tilt_adjust"})
        assert build_probe_method_macros(printer, config) == [
            "BED_MESH_CALIBRATE",
            "Z_TILT_ADJUST",
            "QUAD_GANTRY_LEVEL",
            "SCREWS_TILT_CALCULATE",
        ]


class TestKalicoZTiltAliases:
    """Kalico passes z_tilt_sections=('z_tilt', 'z_tilt_ng')."""

    @pytest.mark.parametrize("sections", [{"z_tilt"}, {"z_tilt_ng"}, {"z_tilt", "z_tilt_ng"}])
    def test_any_z_tilt_variant_adds_exactly_one_z_tilt_adjust(self, sections: set[str]) -> None:
        printer, config = _make_printer_config(sections)
        result = build_probe_method_macros(printer, config, z_tilt_sections=("z_tilt", "z_tilt_ng"))
        assert result.count("Z_TILT_ADJUST") == 1

    def test_z_tilt_ng_full_config_returns_ordered_list(self) -> None:
        printer, config = _make_printer_config({"z_tilt_ng", "quad_gantry_level", "screws_tilt_adjust"})
        assert build_probe_method_macros(printer, config, z_tilt_sections=("z_tilt", "z_tilt_ng")) == [
            "BED_MESH_CALIBRATE",
            "Z_TILT_ADJUST",
            "QUAD_GANTRY_LEVEL",
            "SCREWS_TILT_CALCULATE",
        ]
