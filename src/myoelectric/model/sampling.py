"""Sampling geometry for a discrete time myoelectric record."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SamplingSpec"]


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """Sample rate and record length for one channel.

    A surface electromyogram carries useful power up to roughly 500 Hz, so a sample
    rate of at least 1000 Hz is required to avoid aliasing and 2000 Hz is the usual
    laboratory choice. See De Luca (1997) and Hermens et al. (2000) in the README
    reference list.
    """

    sample_rate_hz: float
    duration_s: float

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.n_samples < 2:
            raise ValueError("the record must contain at least two samples")

    @property
    def n_samples(self) -> int:
        """Number of samples in the record."""
        return round(float(self.sample_rate_hz) * float(self.duration_s))

    @property
    def sample_interval_s(self) -> float:
        """Time between consecutive samples in seconds."""
        return 1.0 / self.sample_rate_hz

    @property
    def nyquist_hz(self) -> float:
        """Highest frequency representable at this sample rate."""
        return 0.5 * self.sample_rate_hz

    def times(self) -> NDArray[np.float64]:
        """Sample times in seconds, starting at zero."""
        return np.arange(self.n_samples, dtype=np.float64) * self.sample_interval_s

    def to_samples(self, seconds: float) -> int:
        """Convert a duration in seconds to a whole number of samples."""
        return round(float(seconds) * float(self.sample_rate_hz))

    def to_seconds(self, samples: int) -> float:
        """Convert a sample count to seconds."""
        return samples * self.sample_interval_s
