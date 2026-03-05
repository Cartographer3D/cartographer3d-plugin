from __future__ import annotations

import pytest
from typing_extensions import final

from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.macros.touch.calibrate import (
    ScreeningResult,
    ThresholdScreener,
    ThresholdVerifier,
    TouchCalibrateMacro,
    VerificationResult,
    calculate_step,
    format_distance,
)
from cartographer.probe.touch_mode import TouchError
from tests.mocks.config import MockConfiguration

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

    def perform_touch_probe(self, *, z_limit: float | None = None) -> float:  # noqa: ARG002
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

    def test_infinity(self):
        assert format_distance(float("inf")) == "inf"

    def test_negative_infinity(self):
        assert format_distance(float("-inf")) == "-inf"

    def test_nan(self):
        assert format_distance(float("nan")) == "nan"


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


# --- Helpers for _find_threshold / _optimize_threshold tests ---


def _make_macro() -> TouchCalibrateMacro:
    """Create a TouchCalibrateMacro with only _config set (for testing private methods)."""
    macro = object.__new__(TouchCalibrateMacro)
    macro._config = MockConfiguration()  # pyright: ignore[reportPrivateUsage]  # samples=5, max_noisy_samples=2
    return macro


@final
class FakeScreener:
    """Fake ThresholdScreener that returns predetermined results.

    Results are returned in order. If exhausted, returns a passing ScreeningResult.
    """

    def __init__(self, results: list[ScreeningResult | None]) -> None:
        self._results = list(results)

    def screen(self, threshold: int, sample_count: int) -> ScreeningResult | None:  # noqa: ARG002
        if self._results:
            return self._results.pop(0)
        # Default: passes screening
        return ScreeningResult(
            threshold=threshold,
            samples=(1.000,) * sample_count,
            best_subset=[1.000] * sample_count,
            best_range=0.001,
        )


@final
class FakeVerifier:
    """Fake ThresholdVerifier that returns predetermined results.

    Results are returned in order. Tracks z_limit values passed.
    """

    def __init__(self, results: list[VerificationResult | None]) -> None:
        self._results = list(results)
        self.z_limits: list[float | None] = []

    def verify(
        self,
        threshold: int,  # noqa: ARG002
        max_verify_range: float,  # noqa: ARG002
        sample_count: int,  # noqa: ARG002
        *,
        z_limit: float | None = None,
    ) -> VerificationResult | None:
        self.z_limits.append(z_limit)
        return self._results.pop(0)


# --- _find_threshold with optimization ---


class TestFindThreshold:
    def test_returns_first_threshold_when_already_optimal(self):
        """When first verification median_range <= sample_range, skip optimization."""
        macro = _make_macro()

        screener = FakeScreener(
            [
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(threshold=1000, probe_medians=[1.000, 1.005, 1.003], median_range=0.005),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result == 1000
        # Verifier should not have been called with z_limit (no optimization)
        assert verifier.z_limits == [None]

    def test_optimization_finds_better_threshold(self):
        """Optimization finds a higher threshold with better accuracy."""
        macro = _make_macro()

        # First pass: threshold 1000 passes verification with 0.015mm range
        # Optimization: threshold 1100 passes with 0.008mm range
        screener = FakeScreener(
            [
                # Initial screening at 1000
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                # Optimization screening at 1100
                ScreeningResult(threshold=1100, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )
        verifier = FakeVerifier(
            [
                # Initial verification at 1000
                VerificationResult(threshold=1000, probe_medians=[1.000, 1.010, 1.015], median_range=0.015),
                # Optimization verification at 1100 — better!
                VerificationResult(threshold=1100, probe_medians=[1.000, 1.004, 1.008], median_range=0.008),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result == 1100
        # First verify: no z_limit; optimization verify: with z_limit
        assert verifier.z_limits[0] is None
        assert verifier.z_limits[1] is not None
        assert verifier.z_limits[1] == pytest.approx(1.000 - 0.020)  # pyright: ignore[reportUnknownMemberType]

    def test_optimization_keeps_first_when_no_improvement(self):
        """If optimization candidates don't improve, keep first threshold."""
        macro = _make_macro()

        screener = FakeScreener(
            [
                # Initial screening at 1000
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                # Optimization screening at 1100
                ScreeningResult(threshold=1100, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )
        verifier = FakeVerifier(
            [
                # Initial verification at 1000 — 0.015mm
                VerificationResult(threshold=1000, probe_medians=[1.000, 1.010, 1.015], median_range=0.015),
                # Optimization verification at 1100 — worse (0.018mm)
                VerificationResult(threshold=1100, probe_medians=[1.000, 1.010, 1.018], median_range=0.018),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result == 1000

    def test_optimization_stops_early_when_definitely_satisfied(self):
        """Optimization exits early when median_range <= sample_range."""
        macro = _make_macro()

        screener = FakeScreener(
            [
                # Initial screening at 1000
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                # Optimization screening at 1100 — first opt candidate
                ScreeningResult(threshold=1100, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                # This should NOT be reached due to early exit
                ScreeningResult(threshold=1200, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )
        verifier = FakeVerifier(
            [
                # Initial verification at 1000 — 0.015mm (not optimal)
                VerificationResult(threshold=1000, probe_medians=[1.000, 1.010, 1.015], median_range=0.015),
                # Optimization at 1100 — 0.008mm (<= 0.010 sample_range → definitely satisfied)
                VerificationResult(threshold=1100, probe_medians=[1.000, 1.004, 1.008], median_range=0.008),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result == 1100
        # Only 2 verify calls — didn't continue after definitely satisfied
        assert len(verifier.z_limits) == 2

    def test_optimization_bounded_by_threshold_max(self):
        """Optimization doesn't exceed threshold_max even if 20% would go higher."""
        macro = _make_macro()

        # threshold 4500, step = 450, first opt at 4950
        # optimization_max = min(int(4500 * 1.2), 5000) = 5000
        # 4950 <= 5000, so one optimization candidate is tried
        screener = FakeScreener(
            [
                ScreeningResult(threshold=4500, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                # Optimization at 4950 passes screening
                ScreeningResult(threshold=4950, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )
        verifier = FakeVerifier(
            [
                # Initial at 4500
                VerificationResult(threshold=4500, probe_medians=[1.000, 1.010, 1.015], median_range=0.015),
                # Optimization at 4950 — slightly worse, so first stays best
                VerificationResult(threshold=4950, probe_medians=[1.000, 1.010, 1.018], median_range=0.018),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=4500,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result == 4500  # No improvement found, original stays

    def test_returns_none_when_no_threshold_passes(self):
        """Returns None when all thresholds fail screening."""
        macro = _make_macro()

        # All screenings fail
        screener = FakeScreener(
            [
                ScreeningResult(threshold=500, samples=(1.0,), best_subset=None, best_range=float("inf")),
                ScreeningResult(threshold=600, samples=(1.0,), best_subset=None, best_range=float("inf")),
                ScreeningResult(threshold=700, samples=(1.0,), best_subset=None, best_range=float("inf")),
                ScreeningResult(threshold=800, samples=(1.0,), best_subset=None, best_range=float("inf")),
                ScreeningResult(threshold=900, samples=(1.0,), best_subset=None, best_range=float("inf")),
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=None, best_range=float("inf")),
            ]
        )
        verifier = FakeVerifier([])

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=500,
            threshold_max=1000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        assert result is None

    def test_optimization_z_limit_computed_correctly(self):
        """z_limit is min(probe_medians) - max_verify_range from first verification."""
        macro = _make_macro()

        screener = FakeScreener(
            [
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
            ]
        )

        probe_medians = [5.100, 5.110, 5.105]  # min = 5.100
        max_verify_range = 0.020
        expected_z_limit = 5.100 - 0.020  # = 5.080

        verifier = FakeVerifier(
            [
                VerificationResult(threshold=1000, probe_medians=probe_medians, median_range=0.015),
                # Optimization candidate — just needs to exist to check z_limit
                VerificationResult(threshold=1100, probe_medians=[5.100, 5.105, 5.108], median_range=0.008),
            ]
        )

        _ = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=max_verify_range,
            verification_samples=3,
        )

        # First verify has no z_limit, optimization verify has computed z_limit
        assert verifier.z_limits[0] is None
        assert verifier.z_limits[1] == pytest.approx(expected_z_limit)  # pyright: ignore[reportUnknownMemberType]

    def test_screening_failure_during_optimization_does_not_crash(self):
        """Screening returning None during optimization is handled gracefully."""
        macro = _make_macro()

        # threshold 1000, step 100, first opt at 1100
        # optimization_max = min(int(1000 * 1.2), 5000) = 1200
        # At 1100: None. Step from None = max(50, int(1100*0.20)) = 220
        # 1100 + 220 = 1320 > 1200, so optimization ends
        screener = FakeScreener(
            [
                ScreeningResult(threshold=1000, samples=(1.0,), best_subset=[1.0], best_range=0.005),
                None,  # Trigger error at 1100 during optimization
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(threshold=1000, probe_medians=[1.000, 1.010, 1.015], median_range=0.015),
            ]
        )

        result = macro._find_threshold(  # pyright: ignore[reportPrivateUsage]
            screener,
            verifier,
            threshold_start=1000,
            threshold_max=5000,
            sample_range=0.010,
            max_verify_range=0.020,
            verification_samples=3,
        )

        # Falls back to first threshold since no improvement was found
        assert result == 1000
