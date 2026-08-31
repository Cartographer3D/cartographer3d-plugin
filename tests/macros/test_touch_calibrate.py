from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import final

from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.macros.fields import parse
from cartographer.macros.touch.calibrate import (
    ScreeningResult,
    ThresholdScreener,
    ThresholdVerifier,
    TouchCalibrateMacro,
    TouchCalibrateParams,
    VerificationResult,
    calculate_step,
    format_distance,
)
from cartographer.probe.touch_mode import TouchError
from tests.mocks.config import MockConfiguration
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

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

    def collect_samples(self, threshold: int, sample_count: int) -> tuple[float, ...]:
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
        assert abs(result.median_range - 0.010) < 1e-9
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


# --- TouchCalibrateParams ---


class TestTouchCalibrateParams:
    def test_fractional_speed_parsed_and_retained(self):
        """CARTOGRAPHER_TOUCH_CALIBRATE SPEED=1.5 must parse to 1.5, not truncated."""
        mock = MockParams()
        mock.params["SPEED"] = "1.5"
        p = parse(
            TouchCalibrateParams,
            mock,
            samples=3,
            sample_range=0.010,
            max_verify_range=0.020,
        )
        assert p.speed == 1.5

    def test_samples_defaults_from_config(self):
        """SAMPLES defaults to the value supplied via config defaults."""
        mock = MockParams()
        p = parse(
            TouchCalibrateParams,
            mock,
            samples=5,
            sample_range=0.010,
            max_verify_range=0.020,
        )
        assert p.samples == 5

    def test_samples_explicit_override(self):
        """Explicit SAMPLES= overrides the config default."""
        mock = MockParams()
        mock.params["SAMPLES"] = "7"
        p = parse(
            TouchCalibrateParams,
            mock,
            samples=5,
            sample_range=0.010,
            max_verify_range=0.020,
        )
        assert p.samples == 7

    def test_sample_range_defaults_from_config(self):
        """SAMPLE_RANGE defaults to the value supplied via config defaults."""
        mock = MockParams()
        p = parse(
            TouchCalibrateParams,
            mock,
            samples=5,
            sample_range=0.008,
            max_verify_range=0.016,
        )
        assert abs(p.sample_range - 0.008) < 1e-9

    def test_sample_range_explicit_override(self):
        """Explicit SAMPLE_RANGE= overrides the config default."""
        mock = MockParams()
        mock.params["SAMPLE_RANGE"] = "0.005"
        p = parse(
            TouchCalibrateParams,
            mock,
            samples=5,
            sample_range=0.010,
            max_verify_range=0.020,
        )
        assert abs(p.sample_range - 0.005) < 1e-9

    def test_max_verify_range_uses_effective_sample_range_default(self, mocker: MockerFixture):
        """MAX_VERIFY_RANGE defaults to 2× effective SAMPLE_RANGE."""
        config = MockConfiguration()
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        # Patch _find_threshold to return a threshold immediately so run() completes.
        find_threshold_spy = mocker.patch.object(macro, "_find_threshold", return_value=1500)
        _ = mocker.patch.object(macro, "_move_to_calibration_position")
        toolhead.is_homed.return_value = True

        params = MockParams()
        # Do not set MAX_VERIFY_RANGE; it should default to 2× config.sample_range (0.010)
        macro.run(params)

        # Prove _find_threshold was called with max_verify_range = 2× effective sample_range
        call_args = find_threshold_spy.call_args
        args = call_args.args
        # args[4] = sample_range, args[5] = max_verify_range
        assert abs(args[4] - config.touch.sample_range) < 1e-9
        assert abs(args[5] - args[4] * 2) < 1e-9

    def test_max_verify_range_scaled_by_explicit_sample_range(self, mocker: MockerFixture):
        """MAX_VERIFY_RANGE defaults to 2× the explicit SAMPLE_RANGE parameter."""
        config = MockConfiguration()
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        find_threshold_spy = mocker.patch.object(macro, "_find_threshold", return_value=1500)
        _ = mocker.patch.object(macro, "_move_to_calibration_position")
        toolhead.is_homed.return_value = True

        params = MockParams()
        params.params["SAMPLE_RANGE"] = "0.005"
        # MAX_VERIFY_RANGE not supplied — should default to 2× 0.005 = 0.010
        macro.run(params)

        # Prove _find_threshold was called with max_verify_range = 2× explicit SAMPLE_RANGE
        call_args = find_threshold_spy.call_args
        args = call_args.args
        # args[4] = sample_range, args[5] = max_verify_range
        assert abs(args[4] - 0.005) < 1e-9
        assert abs(args[5] - args[4] * 2) < 1e-9

    def test_max_verify_range_below_sample_range_raises(self, mocker: MockerFixture):
        """MAX_VERIFY_RANGE below SAMPLE_RANGE must raise RuntimeError."""
        config = MockConfiguration()
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        params = MockParams()
        # sample_range = 0.010 (config default); set max_verify_range below it
        params.params["MAX_VERIFY_RANGE"] = "0.008"

        with pytest.raises(RuntimeError, match="MAX_VERIFY_RANGE"):
            macro.run(params)

    def test_max_verify_range_above_4x_sample_range_raises(self, mocker: MockerFixture):
        """MAX_VERIFY_RANGE above 4× SAMPLE_RANGE must raise RuntimeError."""
        config = MockConfiguration()
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        params = MockParams()
        # sample_range = 0.010 (config default); 4× = 0.040; set above it
        params.params["MAX_VERIFY_RANGE"] = "0.050"

        with pytest.raises(RuntimeError, match="MAX_VERIFY_RANGE"):
            macro.run(params)

    def test_samples_exceeding_max_samples_raises_before_movement(self, mocker: MockerFixture):
        """SAMPLES > config.touch.max_samples must raise RuntimeError before any movement."""
        config = MockConfiguration()  # default: max_samples=10
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        # Patch _move_to_calibration_position to assert it is never called.
        move_spy = mocker.patch.object(macro, "_move_to_calibration_position")

        params = MockParams()
        params.params["SAMPLES"] = "11"  # exceeds max_samples=10

        with pytest.raises(RuntimeError, match="SAMPLES"):
            macro.run(params)

        move_spy.assert_not_called()
        toolhead.move.assert_not_called()

    def test_saved_model_stores_samples_and_sample_range(self, mocker: MockerFixture):
        """Calibration result must persist effective samples and sample_range."""
        config = MockConfiguration()
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        _ = mocker.patch.object(macro, "_find_threshold", return_value=2000)
        _ = mocker.patch.object(macro, "_move_to_calibration_position")
        toolhead.is_homed.return_value = True

        params = MockParams()
        params.params["SAMPLES"] = "7"
        params.params["SAMPLE_RANGE"] = "0.008"
        macro.run(params)

        saved = config.touch.models["default"]
        assert saved.samples == 7
        assert saved.sample_range is not None
        assert abs(saved.sample_range - 0.008) < 1e-9

    def test_calibration_execution_uses_effective_samples(self, mocker: MockerFixture):
        """The effective SAMPLES value drives screening_samples passed to _find_threshold."""
        config = MockConfiguration()  # default: samples=5, max_noisy_samples=2
        probe = mocker.Mock()
        mcu = mocker.Mock()
        toolhead = mocker.Mock()
        task_executor = mocker.Mock()
        macro = TouchCalibrateMacro(probe, mcu, toolhead, config, task_executor)

        find_threshold_spy = mocker.patch.object(macro, "_find_threshold", return_value=1500)
        _ = mocker.patch.object(macro, "_move_to_calibration_position")
        toolhead.is_homed.return_value = True

        params = MockParams()
        params.params["SAMPLES"] = "4"  # override config default of 5
        macro.run(params)

        # _find_threshold receives screening_samples = SAMPLES + max_noisy_samples = 4 + 2 = 6
        call_args = find_threshold_spy.call_args
        # positional args: screener, verifier, start, max, sample_range,
        # max_verify, verify_samples, screening_samples
        args = call_args.args
        assert args[7] == 6  # screening_samples at position index 7
