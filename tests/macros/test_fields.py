"""Integration tests: ComputedDefault metadata rendered through generate_docs()."""

from __future__ import annotations

import pytest

from cartographer.macros.docs import generate_docs
from cartographer.macros.fields import ComputedDefault, get_all_params
from cartographer.macros.touch.calibrate import TouchCalibrateParams

_EXPECTED_COMPUTED_TEXT = "computed: 2 \u00d7 effective SAMPLE_RANGE"
_MACRO_HEADER = "## CARTOGRAPHER_TOUCH_CALIBRATE"


def _touch_calibrate_section() -> str:
    """Extract the CARTOGRAPHER_TOUCH_CALIBRATE section from generated docs."""
    docs = generate_docs()
    start = docs.find(_MACRO_HEADER)
    assert start != -1, f"{_MACRO_HEADER!r} section not found in generated docs"
    # Find the next ## heading after this section
    next_section = docs.find("\n## ", start + len(_MACRO_HEADER))
    return docs[start:] if next_section == -1 else docs[start:next_section]


@pytest.fixture(scope="module")
def calibrate_section() -> str:
    return _touch_calibrate_section()


class TestMaxVerifyRangeDocs:
    """MAX_VERIFY_RANGE must render as computed, optional, and absent from example."""

    def test_docs_show_computed_default(self, calibrate_section: str) -> None:
        """Generated docs must label MAX_VERIFY_RANGE with its computed formula."""
        assert _EXPECTED_COMPUTED_TEXT in calibrate_section, (
            f"Expected {_EXPECTED_COMPUTED_TEXT!r} in CARTOGRAPHER_TOUCH_CALIBRATE section"
        )

    def test_docs_do_not_mark_required(self, calibrate_section: str) -> None:
        """MAX_VERIFY_RANGE must not appear as a required parameter."""
        # The required pattern is "MAX_VERIFY_RANGE (float, required)"
        assert "MAX_VERIFY_RANGE (float, required)" not in calibrate_section

    def test_example_omits_max_verify_range(self, calibrate_section: str) -> None:
        """Example command line must not contain a MAX_VERIFY_RANGE placeholder."""
        # Locate the Example block inside the section
        example_start = calibrate_section.find("**Example:**")
        assert example_start != -1, "Example block not found in CARTOGRAPHER_TOUCH_CALIBRATE section"
        example_block = calibrate_section[example_start:]
        assert "MAX_VERIFY_RANGE" not in example_block

    def test_max_verify_range_metadata_is_computed_default(self) -> None:
        """Direct metadata check: MAX_VERIFY_RANGE must carry a ComputedDefault sentinel."""
        params = get_all_params(TouchCalibrateParams)
        field = next((p for p in params if p.name == "MAX_VERIFY_RANGE"), None)
        assert field is not None, "MAX_VERIFY_RANGE not found in TouchCalibrateParams"
        assert isinstance(field.default, ComputedDefault), (
            f"Expected ComputedDefault, got {type(field.default).__name__}"
        )
        assert "2 × effective SAMPLE_RANGE" in field.default.display

    def test_docs_show_valid_interval(self, calibrate_section: str) -> None:
        """Generated docs must state the valid interval relative to SAMPLE_RANGE."""
        assert "at least effective SAMPLE_RANGE" in calibrate_section
        assert "4 \u00d7 effective SAMPLE_RANGE" in calibrate_section

    def test_docs_show_samples_max_samples_constraint(self, calibrate_section: str) -> None:
        """Generated docs must state SAMPLES cannot exceed max_samples."""
        assert "Cannot exceed configured [cartographer touch] max_samples" in calibrate_section
