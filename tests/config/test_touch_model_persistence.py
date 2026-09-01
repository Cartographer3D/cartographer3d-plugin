from __future__ import annotations

from cartographer.interfaces.configuration import TouchModelConfiguration


class TestTouchModelFields:
    def test_model_has_concrete_types(self) -> None:
        """TouchModelConfiguration fields are concrete (not Optional) int/float."""
        cfg = TouchModelConfiguration(name="model", speed=3.0, z_offset=0.0, samples=3, sample_range=0.010)
        assert isinstance(cfg.samples, int)
        assert isinstance(cfg.sample_range, float)

    def test_non_default_values_stored(self) -> None:
        """Non-default samples/sample_range values are preserved as-is."""
        cfg = TouchModelConfiguration(name="custom", speed=3.0, z_offset=0.0, samples=7, sample_range=0.008)
        assert cfg.samples == 7
        assert abs(cfg.sample_range - 0.008) < 1e-9
