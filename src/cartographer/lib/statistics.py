from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def compute_mad(samples: NDArray[np.float_] | Sequence[float]) -> float:
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 1:
        return float("inf")
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))
    return float(mad)
