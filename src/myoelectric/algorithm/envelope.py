"""Amplitude estimation for proportional myoelectric control.

A proportional controller drives a prosthesis at a speed set by the amplitude of the
myoelectric signal, so the quality of the amplitude estimate is the quality of the
control. Every estimator below trades the same two quantities against each other. More
smoothing gives a steadier command, which the user perceives as controllable, and a
longer delay between the muscle contracting and the device moving, which the user
perceives as unresponsive. Farrell and Weir (2007) measured the optimal controller
delay for myoelectric prostheses and reported a usable range with an upper bound near
100 ms to 125 ms of total delay, of which the amplitude estimator is only one part.

Four estimators are provided, all causal:

Moving average of the rectified signal
    The mean absolute value over a trailing window of ``n`` samples. The window is a
    finite impulse response filter with linear phase, so its group delay is exactly
    ``(n - 1) / 2`` samples at every frequency.

Moving root mean square
    The square root of the mean square over a trailing window. Same delay as the moving
    average by the same argument, since the averaging kernel is the same.

Low pass filtered rectified signal
    A Butterworth low pass applied causally to the rectified signal. Its group delay
    depends on frequency; the value quoted as nominal is the group delay at zero
    frequency, which is what a step in contraction level experiences.

Exponential moving average
    A single pole recursion ``y[n] = a x[n] + (1 - a) y[n-1]``, the cheapest estimator
    to run on an embedded controller. Its group delay at zero frequency is
    ``(1 - a) / a`` samples.

The nominal delay above is a design figure. :func:`measure_latency` measures the delay
that an estimator actually imposes on a step change in contraction level, and it is the
measured value that is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.filters import apply_causal, design_lowpass, group_delay_samples

__all__ = [
    "EnvelopeEstimator",
    "ExponentialEnvelope",
    "LatencyMeasurement",
    "LowPassEnvelope",
    "MovingAverageEnvelope",
    "MovingRmsEnvelope",
    "measure_latency",
]


@runtime_checkable
class EnvelopeEstimator(Protocol):
    """Common interface for causal amplitude estimators."""

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        ...

    def nominal_delay_samples(self, sample_rate_hz: float) -> float:
        """Design group delay at zero frequency, in samples."""
        ...

    def estimate(self, x: NDArray[np.float64], sample_rate_hz: float) -> NDArray[np.float64]:
        """Amplitude estimate on the same sample grid as ``x``."""
        ...


def _trailing_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.asarray(np.convolve(values, kernel)[: values.size], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MovingAverageEnvelope:
    """Mean absolute value over a trailing window."""

    window_s: float = 0.100

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return f"moving-average-{self.window_s * 1e3:.0f}ms"

    def window_samples(self, sample_rate_hz: float) -> int:
        """Window length in samples, at least two."""
        return max(2, round(float(self.window_s) * float(sample_rate_hz)))

    def nominal_delay_samples(self, sample_rate_hz: float) -> float:
        """Group delay of a rectangular kernel, ``(n - 1) / 2`` samples."""
        return 0.5 * (self.window_samples(sample_rate_hz) - 1)

    def estimate(self, x: NDArray[np.float64], sample_rate_hz: float) -> NDArray[np.float64]:
        """Amplitude estimate on the same sample grid as ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        return _trailing_mean(np.abs(samples), self.window_samples(sample_rate_hz))


@dataclass(frozen=True, slots=True)
class MovingRmsEnvelope:
    """Root mean square over a trailing window."""

    window_s: float = 0.100

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return f"moving-rms-{self.window_s * 1e3:.0f}ms"

    def window_samples(self, sample_rate_hz: float) -> int:
        """Window length in samples, at least two."""
        return max(2, round(float(self.window_s) * float(sample_rate_hz)))

    def nominal_delay_samples(self, sample_rate_hz: float) -> float:
        """Group delay of a rectangular kernel, ``(n - 1) / 2`` samples."""
        return 0.5 * (self.window_samples(sample_rate_hz) - 1)

    def estimate(self, x: NDArray[np.float64], sample_rate_hz: float) -> NDArray[np.float64]:
        """Amplitude estimate on the same sample grid as ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        mean_square = _trailing_mean(samples**2, self.window_samples(sample_rate_hz))
        return np.asarray(np.sqrt(np.maximum(mean_square, 0.0)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LowPassEnvelope:
    """Butterworth low pass applied causally to the rectified signal."""

    cutoff_hz: float = 4.0
    order: int = 2

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return f"lowpass-{self.cutoff_hz:g}Hz-order{self.order}"

    def nominal_delay_samples(self, sample_rate_hz: float) -> float:
        """Group delay of the design at zero frequency, in samples."""
        design = design_lowpass(sample_rate_hz, self.cutoff_hz, order=self.order)
        return float(group_delay_samples(design, np.array([0.0]), mode="causal")[0])

    def estimate(self, x: NDArray[np.float64], sample_rate_hz: float) -> NDArray[np.float64]:
        """Amplitude estimate on the same sample grid as ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        design = design_lowpass(sample_rate_hz, self.cutoff_hz, order=self.order)
        return np.asarray(np.maximum(apply_causal(design, np.abs(samples)), 0.0))


@dataclass(frozen=True, slots=True)
class ExponentialEnvelope:
    """Single pole exponential moving average of the rectified signal."""

    time_constant_s: float = 0.050

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return f"exponential-{self.time_constant_s * 1e3:.0f}ms"

    def alpha(self, sample_rate_hz: float) -> float:
        """Recursion coefficient for the requested time constant."""
        return float(1.0 - np.exp(-1.0 / (self.time_constant_s * sample_rate_hz)))

    def nominal_delay_samples(self, sample_rate_hz: float) -> float:
        """Group delay at zero frequency, ``(1 - a) / a`` samples."""
        a = self.alpha(sample_rate_hz)
        return (1.0 - a) / a

    def estimate(self, x: NDArray[np.float64], sample_rate_hz: float) -> NDArray[np.float64]:
        """Amplitude estimate on the same sample grid as ``x``."""
        samples = np.abs(np.asarray(x, dtype=np.float64).ravel())
        a = self.alpha(sample_rate_hz)
        output = np.empty_like(samples)
        state = float(samples[0])
        for index, value in enumerate(samples):
            state += a * (float(value) - state)
            output[index] = state
        return output


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    """Measured step response of one amplitude estimator."""

    estimator: str
    nominal_delay_ms: float
    latency_ms: float
    rise_time_ms: float
    plateau_ripple_percent: float

    def as_row(self) -> tuple[str, float, float, float, float]:
        """Fields in table order, for reporting."""
        return (
            self.estimator,
            self.nominal_delay_ms,
            self.latency_ms,
            self.rise_time_ms,
            self.plateau_ripple_percent,
        )


def _crossing_time_s(
    envelope: NDArray[np.float64],
    start_index: int,
    level: float,
    sample_rate_hz: float,
) -> float:
    """Time from ``start_index`` to the first sample at or above ``level``, in seconds.

    Sub sample resolution is obtained by linear interpolation between the bracketing
    samples, so the estimate is not quantised to the sample interval.
    """
    tail = envelope[start_index:]
    above = np.flatnonzero(tail >= level)
    if above.size == 0:
        return float("nan")
    index = int(above[0])
    if index == 0:
        return 0.0
    lower = float(tail[index - 1])
    upper = float(tail[index])
    fraction = 0.0 if upper == lower else (level - lower) / (upper - lower)
    return (index - 1 + fraction) / sample_rate_hz


def measure_latency(
    estimator: EnvelopeEstimator,
    x: NDArray[np.float64],
    sample_rate_hz: float,
    step_index: int,
    plateau_slice: slice,
) -> LatencyMeasurement:
    """Measure the delay and the residual ripple that ``estimator`` imposes.

    The record ``x`` must contain a step change in contraction level at ``step_index``,
    a resting segment before it, and a steady contraction after it. Three quantities
    are measured.

    Latency
        Time from the step to the instant the estimate first reaches half of the total
        change between the resting level and the plateau level. The half amplitude
        point is used because it is where the estimate is steepest and therefore where
        the crossing time is least sensitive to residual ripple.

    Rise time
        Time from the ten per cent crossing to the ninety per cent crossing.

    Plateau ripple
        Standard deviation of the estimate over ``plateau_slice``, divided by its mean,
        in per cent. This is the quantity that smoothing buys, expressed as the
        variability that a proportional controller would pass to the actuator.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    envelope = estimator.estimate(samples, sample_rate_hz)
    # The resting level is taken from the second half of the pre step segment. Every
    # estimator here starts from a zero state, so the first part of that segment holds
    # the estimator's own startup transient rather than the resting level, and including
    # it would pull the half amplitude threshold down and report a latency that is too
    # short by a fraction of the window.
    rest_start = max(0, step_index // 2)
    rest = envelope[rest_start : max(rest_start + 2, step_index)]
    rest_level = float(np.mean(rest))
    plateau = envelope[plateau_slice]
    if plateau.size < 2:
        raise ValueError("plateau_slice must select at least two samples")
    plateau_level = float(np.mean(plateau))
    span = plateau_level - rest_level
    if span <= 0.0:
        raise ValueError("the plateau level must exceed the resting level")

    half = _crossing_time_s(envelope, step_index, rest_level + 0.5 * span, sample_rate_hz)
    low = _crossing_time_s(envelope, step_index, rest_level + 0.1 * span, sample_rate_hz)
    high = _crossing_time_s(envelope, step_index, rest_level + 0.9 * span, sample_rate_hz)
    ripple = 100.0 * float(np.std(plateau)) / plateau_level

    return LatencyMeasurement(
        estimator=estimator.name,
        nominal_delay_ms=1e3 * estimator.nominal_delay_samples(sample_rate_hz) / sample_rate_hz,
        latency_ms=1e3 * half,
        rise_time_ms=1e3 * (high - low),
        plateau_ripple_percent=ripple,
    )
