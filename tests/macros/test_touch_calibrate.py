from __future__ import annotations

import pytest
from typing_extensions import final

from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.macros.touch.calibrate import (
    ScreeningResult,
    ThresholdScreener,
    ThresholdVerifier,
    VerificationResult,
    calculate_step,
    format_distance,
)
from cartographer.probe.touch_mode import TouchError

# --- Fake probe for testing ---


@final
class FakeCalibrationProbe:
    """Fake implementation of CalibrationProbe for testing.

    Configure with sequences of return values. Each call to
    collect_samples / perform_touch_probe pops the next value.
    If the value is a ProbeTriggerError, it is raised instead.
    """

    def __init__(
        self,
        *,
        samples_results: list[tuple[float, ...] | ProbeTriggerError] | None = None,
        probe_results: list[float | RuntimeError] | None = None,
    ) -> None:
        self._samples_results = list(samples_results or [])
        self._probe_results = list(probe_results or [])
        self.thresholds_set: list[int] = []

    def collect_samples(self, threshold: int, sample_count: int) -> tuple[float, ...]:  # noqa: ARG002
        _ = threshold, sample_count
        result = self._samples_results.pop(0)
        if isinstance(result, ProbeTriggerError):
            raise result
        return result

    def set_threshold(self, threshold: int) -> None:
        self.thresholds_set.append(threshold)

    def perform_touch_probe(self) -> float:
        result = self._probe_results.pop(0)
        if isinstance(result, RuntimeError):
            raise result
        return result


# --- Data classes ---


class TestScreeningResult:
    def test_passes_when_range_within_limit(self):
        result = ScreeningResult(
            threshold=1000,
            samples=(1.0, 1.005, 1.008),
            best_subset=[1.0, 1.005, 1.008],
            best_range=0.008,
        )
        assert result.passed(sample_range=0.010)

    def test_fails_when_range_exceeds_limit(self):
        result = ScreeningResult(
            threshold=1000,
            samples=(1.0, 1.005, 1.020),
            best_subset=[1.0, 1.005, 1.020],
            best_range=0.020,
        )
        assert not result.passed(sample_range=0.010)

    def test_fails_when_range_equals_infinity(self):
        result = ScreeningResult(
            threshold=1000,
            samples=(1.0,),
            best_subset=None,
            best_range=float("inf"),
        )
        assert not result.passed(sample_range=0.010)


class TestVerificationResult:
    def test_passes_when_medians_consistent(self):
        result = VerificationResult(
            threshold=1000,
            probe_medians=[1.000, 1.005, 1.008, 1.003, 1.006],
            median_range=0.008,
        )
        assert result.passed(max_verify_range=0.020)

    def test_fails_when_medians_inconsistent(self):
        result = VerificationResult(
            threshold=1000,
            probe_medians=[1.000, 1.050, 1.030],
            median_range=0.050,
        )
        assert not result.passed(max_verify_range=0.020)


# --- format_distance ---


class TestFormatDistance:
    def test_rounds_with_ceiling(self):
        # 0.0001 -> ceil(0.1) / 1000 = 0.001
        assert format_distance(0.0001) == "0.001"

    def test_exact_value(self):
        assert format_distance(0.010) == "0.010"

    def test_zero(self):
        assert format_distance(0.0) == "0.000"


# --- ThresholdScreener ---


class TestThresholdScreener:
    def test_returns_none_on_trigger_error(self):
        probe = FakeCalibrationProbe(samples_results=[ProbeTriggerError("triggered")])
        screener = ThresholdScreener(probe, required_samples=3)

        result = screener.screen(threshold=1000, sample_count=5)

        assert result is None

    def test_passes_with_tight_samples(self):
        probe = FakeCalibrationProbe(
            samples_results=[(1.000, 1.002, 1.004, 1.006, 1.008)],
        )
        screener = ThresholdScreener(probe, required_samples=3)

        result = screener.screen(threshold=1000, sample_count=5)

        assert result is not None
        assert result.passed(sample_range=0.010)
        assert result.threshold == 1000

    def test_fails_with_spread_samples(self):
        probe = FakeCalibrationProbe(
            samples_results=[(1.000, 1.050, 1.100, 1.150, 1.200)],
        )
        screener = ThresholdScreener(probe, required_samples=3)

        result = screener.screen(threshold=1000, sample_count=5)

        assert result is not None
        assert not result.passed(sample_range=0.010)

    def test_returns_all_samples_in_result(self):
        samples = (1.000, 1.002, 1.004, 1.006, 1.008)
        probe = FakeCalibrationProbe(samples_results=[samples])
        screener = ThresholdScreener(probe, required_samples=3)

        result = screener.screen(threshold=2000, sample_count=5)

        assert result is not None
        assert result.samples == samples
        assert result.threshold == 2000


# --- ThresholdVerifier ---


class TestThresholdVerifier:
    def test_passes_with_consistent_probes(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.005, 1.003, 1.007, 1.002],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is not None
        assert result.passed(max_verify_range=0.020)
        assert len(result.probe_medians) == 5

    def test_fails_with_inconsistent_probes(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.100, 1.050, 1.200, 1.010],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is not None
        assert not result.passed(max_verify_range=0.020)

    def test_exits_early_when_inconsistent(self):
        # Provide more results than needed — verifier should stop early
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.100, 1.200, 1.300, 1.400],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is not None
        # Should have stopped before running all 5 probes
        assert len(result.probe_medians) < 5
        assert not result.passed(max_verify_range=0.020)

    def test_returns_none_on_trigger_error(self):
        probe = FakeCalibrationProbe(
            probe_results=[ProbeTriggerError("triggered")],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is None

    def test_returns_none_on_mid_sequence_trigger_error(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.003, ProbeTriggerError("triggered")],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is None

    def test_returns_none_on_touch_error(self):
        """TouchError (e.g. unable to find consistent samples) returns None."""
        probe = FakeCalibrationProbe(
            probe_results=[TouchError("Unable to find 3 samples within 0.010mm")],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is None

    def test_returns_none_on_mid_sequence_touch_error(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.003, TouchError("Unable to find consistent samples")],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=5)

        assert result is None

    def test_sets_threshold_before_probing(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.003, 1.005],
        )
        verifier = ThresholdVerifier(probe)

        _ = verifier.verify(threshold=2500, max_verify_range=0.020, sample_count=3)

        assert probe.thresholds_set == [2500]

    def test_median_range_calculated_correctly(self):
        probe = FakeCalibrationProbe(
            probe_results=[1.000, 1.010, 1.005],
        )
        verifier = ThresholdVerifier(probe)

        result = verifier.verify(threshold=1000, max_verify_range=0.020, sample_count=3)

        assert result is not None
        assert result.median_range == pytest.approx(0.010)  # pyright: ignore[reportUnknownMemberType]
        assert result.probe_medians == [1.000, 1.010, 1.005]


# --- calculate_step ---


class TestCalculateStep:
    def test_large_step_when_range_is_none(self):
        """No range info (trigger error) uses 20% of threshold."""
        step = calculate_step(threshold=1000, range_value=None, sample_range=0.010)
        assert step == 200  # 1000 * 0.20

    def test_large_step_when_range_is_very_bad(self):
        """Range > 10x sample_range uses 20% of threshold."""
        step = calculate_step(threshold=1000, range_value=0.200, sample_range=0.010)
        assert step == 200  # 1000 * 0.20

    def test_small_step_when_range_is_close(self):
        """Range near target uses 10% of threshold."""
        step = calculate_step(threshold=1000, range_value=0.015, sample_range=0.010)
        assert step == 100  # 1000 * 0.10

    def test_step_clamped_to_min(self):
        """Step never goes below MIN_STEP (50)."""
        step = calculate_step(threshold=100, range_value=0.015, sample_range=0.010)
        assert step == 50  # 100 * 0.10 = 10, clamped to 50

    def test_step_clamped_to_max(self):
        """Step never goes above MAX_STEP (1000)."""
        step = calculate_step(threshold=10000, range_value=None, sample_range=0.010)
        assert step == 1000  # 10000 * 0.20 = 2000, clamped to 1000

    def test_boundary_at_10x_sample_range(self):
        """Range exactly at 10x boundary uses small step (not strictly greater)."""
        step = calculate_step(threshold=1000, range_value=0.100, sample_range=0.010)
        assert step == 100  # 0.100 is not > 0.100, so uses 10%

    def test_just_above_10x_sample_range(self):
        """Range just above 10x uses large step."""
        step = calculate_step(threshold=1000, range_value=0.101, sample_range=0.010)
        assert step == 200  # 0.101 > 10 * 0.010, uses 20%
