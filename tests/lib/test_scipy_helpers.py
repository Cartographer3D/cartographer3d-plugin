from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cartographer.lib.scipy_helpers import (
    is_available,
    raise_if_curve_fit_unavailable,
    raise_if_rbf_interpolator_unavailable,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestIsAvailable:
    """Tests for is_available() function."""

    def test_is_available_true(self) -> None:
        """is_available() returns True when scipy is importable."""
        result = is_available()
        assert result is True

    def test_is_available_false_not_installed(self) -> None:
        """is_available() returns False when scipy.optimize is not importable."""
        with patch.dict(sys.modules, {"scipy.optimize": None}):
            result = is_available()
        assert result is False

    def test_is_available_false_import_error(self) -> None:
        """is_available() returns False when scipy.optimize raises ImportError."""
        with patch.dict(sys.modules, {"scipy.optimize": None}):
            result = is_available()
        assert result is False


class TestRaiseIfCurveFitUnavailable:
    """Tests for raise_if_curve_fit_unavailable() function."""

    def test_not_installed(self, mocker: MockerFixture) -> None:
        """raise_if_curve_fit_unavailable() raises RuntimeError with 'not installed' when find_spec returns None."""
        _ = mocker.patch("cartographer.lib.scipy_helpers.find_spec", return_value=None)
        with pytest.raises(RuntimeError, match="not installed"):
            raise_if_curve_fit_unavailable()

    def test_import_error(self, mocker: MockerFixture) -> None:
        """raise_if_curve_fit_unavailable() raises RuntimeError with 'failed to import' when import fails."""
        _ = mocker.patch("cartographer.lib.scipy_helpers.find_spec", return_value="mock_spec")
        _ = mocker.patch(
            "cartographer.lib.scipy_helpers.import_module",
            side_effect=ImportError("no module"),
        )
        with pytest.raises(RuntimeError, match="failed to import"):
            raise_if_curve_fit_unavailable()

    def test_success(self) -> None:
        """raise_if_curve_fit_unavailable() does not raise when scipy is available."""
        raise_if_curve_fit_unavailable()


class TestRaiseIfRbfInterpolatorUnavailable:
    """Tests for raise_if_rbf_interpolator_unavailable() function."""

    def test_not_installed(self, mocker: MockerFixture) -> None:
        """Raises RuntimeError with 'not installed' when find_spec returns None."""
        _ = mocker.patch("cartographer.lib.scipy_helpers.find_spec", return_value=None)
        with pytest.raises(RuntimeError, match="not installed"):
            raise_if_rbf_interpolator_unavailable()

    def test_import_error(self, mocker: MockerFixture) -> None:
        """raise_if_rbf_interpolator_unavailable() raises RuntimeError with 'failed to import' when import fails."""
        _ = mocker.patch("cartographer.lib.scipy_helpers.find_spec", return_value="mock_spec")
        _ = mocker.patch(
            "cartographer.lib.scipy_helpers.import_module",
            side_effect=ImportError("no module"),
        )
        with pytest.raises(RuntimeError, match="failed to import"):
            raise_if_rbf_interpolator_unavailable()

    def test_success(self) -> None:
        """raise_if_rbf_interpolator_unavailable() does not raise when scipy is available."""
        raise_if_rbf_interpolator_unavailable()
