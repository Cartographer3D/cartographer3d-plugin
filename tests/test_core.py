from dataclasses import replace
from unittest.mock import Mock

import pytest

from cartographer.core import PrinterCartographer
from cartographer.interfaces.configuration import Configuration
from cartographer.macros.message import MessageMacro
from cartographer.runtime.adapters import Adapters


@pytest.fixture
def mock_adapters(config: Configuration):
    """Create mock adapters for testing."""
    adapters = Mock()
    adapters.mcu = Mock()
    adapters.config = config
    adapters.axis_twist_compensation = None
    adapters.toolhead = Mock()
    adapters.bed_mesh = Mock()
    adapters.task_executor = Mock()
    adapters.gcode = Mock()

    # Add other necessary mock configurations
    return adapters


class TestMacroRegistration:
    """Test that all expected macros are registered correctly."""

    def test_core_probe_macros_registered(self, mock_adapters: Adapters):
        """Verify all core probe macros are registered."""
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        expected = {"PROBE", "PROBE_ACCURACY", "QUERY_PROBE", "Z_OFFSET_APPLY_PROBE"}
        assert expected.issubset(registered_names), f"Missing core probe macros: {expected - registered_names}"

    def test_cartographer_prefixed_macros_registered(self, mock_adapters: Adapters):
        """Verify all CARTOGRAPHER_ prefixed macros are registered."""
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        expected_base_names = {
            "QUERY",
            "STREAM",
            "TEMPERATURE_CALIBRATE",
            "SCAN_CALIBRATE",
            "SCAN_ACCURACY",
            "SCAN_MODEL",
            "ESTIMATE_BACKLASH",
            "TOUCH_CALIBRATE",
            "TOUCH_MODEL",
            "TOUCH_PROBE",
            "TOUCH_ACCURACY",
            "TOUCH_HOME",
        }

        expected_prefixed = {f"CARTOGRAPHER_{name}" for name in expected_base_names}
        assert expected_prefixed.issubset(registered_names), (
            f"Missing CARTOGRAPHER_ prefixed macros: {expected_prefixed - registered_names}"
        )

    def test_bed_mesh_calibrate_registered(self, mock_adapters: Adapters):
        """Verify BED_MESH_CALIBRATE is registered without prefix."""
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        assert "BED_MESH_CALIBRATE" in registered_names
        assert "CARTOGRAPHER_BED_MESH_CALIBRATE" not in registered_names

    def test_legacy_macros_registered(self, mock_adapters: Adapters):
        """Verify legacy macro aliases are registered."""
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        legacy = {
            "CARTOGRAPHER_TOUCH",
            "CARTOGRAPHER_CALIBRATE",
            "CARTOGRAPHER_THRESHOLD_SCAN",
        }

        assert legacy.issubset(registered_names), f"Missing legacy macros: {legacy - registered_names}"

    def test_axis_twist_compensation_with_adapter(self, mock_adapters: Adapters):
        """Verify AXIS_TWIST_COMPENSATION registered when adapter present."""
        mock_adapters.axis_twist_compensation = Mock()
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        assert "CARTOGRAPHER_AXIS_TWIST_COMPENSATION" in registered_names

        # Verify it's the actual macro, not a message
        atc_reg = next(reg for reg in cartographer.macros if reg.name == "CARTOGRAPHER_AXIS_TWIST_COMPENSATION")
        assert not isinstance(atc_reg.macro, MessageMacro)

    def test_axis_twist_compensation_without_adapter(self, mock_adapters: Adapters):
        """Verify AXIS_TWIST_COMPENSATION is message when adapter absent."""
        mock_adapters.axis_twist_compensation = None
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        assert "CARTOGRAPHER_AXIS_TWIST_COMPENSATION" in registered_names

        # Verify it's a message macro
        atc_reg = next(reg for reg in cartographer.macros if reg.name == "CARTOGRAPHER_AXIS_TWIST_COMPENSATION")
        from cartographer.macros.message import MessageMacro

        assert isinstance(atc_reg.macro, MessageMacro)

    def test_custom_prefix_macros(self, mock_adapters: Adapters):
        """Verify custom prefix is applied correctly."""
        mock_adapters.config.general = replace(mock_adapters.config.general, macro_prefix="CUSTOM")
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = {reg.name for reg in cartographer.macros}

        # Should have both CARTOGRAPHER_ and CUSTOM_ versions
        assert "CARTOGRAPHER_QUERY" in registered_names
        assert "CUSTOM_QUERY" in registered_names
        assert "CUSTOM_SCAN_CALIBRATE" in registered_names

    def test_no_duplicate_registrations(self, mock_adapters: Adapters):
        """Verify no macros are registered twice."""
        cartographer = PrinterCartographer(mock_adapters)
        registered_names = [reg.name for reg in cartographer.macros]

        # Check for duplicates
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in registered_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)

        assert not duplicates, f"Duplicate macro registrations: {duplicates}"
