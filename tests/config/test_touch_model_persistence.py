from __future__ import annotations

from cartographer.interfaces.configuration import TouchModelConfiguration


class TestTouchModelDefaults:
    def test_legacy_model_samples_default_is_none(self) -> None:
        """TouchModelConfiguration.samples defaults to None (legacy models have no override)."""
        cfg = TouchModelConfiguration(name="legacy", speed=3.0, z_offset=0.0)
        assert cfg.samples is None

    def test_legacy_model_sample_range_default_is_none(self) -> None:
        """TouchModelConfiguration.sample_range defaults to None (legacy models have no override)."""
        cfg = TouchModelConfiguration(name="legacy", speed=3.0, z_offset=0.0)
        assert cfg.sample_range is None


class TestTouchModelPopulatedFields:
    def test_samples_stored_when_set(self) -> None:
        """A configured samples value is stored and retrievable."""
        cfg = TouchModelConfiguration(name="custom", speed=3.0, z_offset=0.0, samples=5)
        assert cfg.samples == 5

    def test_sample_range_stored_when_set(self) -> None:
        """A configured sample_range value is stored and retrievable."""
        cfg = TouchModelConfiguration(name="ranged", speed=3.0, z_offset=0.0, sample_range=0.008)
        assert cfg.sample_range is not None
        assert abs(cfg.sample_range - 0.008) < 1e-9

    def test_both_fields_stored_independently(self) -> None:
        """Both samples and sample_range can be set and retrieved independently."""
        cfg = TouchModelConfiguration(name="both", speed=3.0, z_offset=0.0, samples=7, sample_range=0.012)
        assert cfg.samples == 7
        assert cfg.sample_range is not None
        assert abs(cfg.sample_range - 0.012) < 1e-9

    def test_samples_none_not_overwritten_by_default(self) -> None:
        """Round-trip: a model created without samples keeps samples=None."""
        original = TouchModelConfiguration(name="rt_legacy", speed=3.0, z_offset=0.0)
        assert original.samples is None
        assert original.sample_range is None
