"""Time domain feature library for myoelectric signals.

Every definition below is written out explicitly so that a reader can verify the
implementation against the source without running it. The five features introduced by
Hudgins, Parker and Scott (1993) as the standard myoelectric control set are mean
absolute value, mean absolute value slope, zero crossings, slope sign changes and
waveform length. Root mean square, variance, integrated electromyogram and Willison
amplitude follow the definitions collected by Phinyomark, Phukpattaranont and Limsakul
(2012). Autoregressive coefficients follow Graupe and Cline (1975).

For a window ``x[0] ... x[N-1]``:

============================  ==================================================
Feature                       Definition
============================  ==================================================
Mean absolute value           ``(1/N) sum |x[i]|``
Mean absolute value slope     ``MAV(segment k+1) - MAV(segment k)``
Zero crossings                count of ``i`` with ``x[i] x[i+1] < 0`` and
                              ``|x[i] - x[i+1]| >= threshold``
Slope sign changes            count of ``i`` in ``1 ... N-2`` with
                              ``(x[i] - x[i-1]) (x[i] - x[i+1]) >= threshold``
Waveform length               ``sum |x[i+1] - x[i]|``
Root mean square              ``sqrt((1/N) sum x[i]^2)``
Variance                      ``(1/(N-1)) sum x[i]^2``
Integrated electromyogram     ``sum |x[i]|``
Willison amplitude            count of ``i`` with ``|x[i] - x[i+1]| > threshold``
Autoregressive coefficients   Yule Walker solution of order ``p``
============================  ==================================================

The variance definition assumes a zero mean signal, which is the convention in the
myoelectric feature literature and is why the sample mean is not subtracted. Surface
recordings are high pass filtered before feature extraction, so the assumption holds.

Threshold behaviour under a change of scale. Multiplying the signal by ``g`` multiplies
mean absolute value, root mean square, integrated electromyogram and waveform length by
``g``, and multiplies variance by ``g^2``. Zero crossings, slope sign changes and
Willison amplitude are unchanged provided their thresholds are scaled by the same
factor as the quantity they are compared against: by ``g`` for zero crossings and
Willison amplitude, which compare amplitudes, and by ``g^2`` for slope sign changes,
which compares a product of two first differences. Autoregressive coefficients are
scale invariant because they describe the correlation structure rather than the
amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.autoregressive import yule_walker

__all__ = [
    "TimeDomainFeatures",
    "autoregressive_coefficients",
    "integrated_emg",
    "mean_absolute_value",
    "mean_absolute_value_slope",
    "root_mean_square",
    "slope_sign_changes",
    "time_domain_features",
    "variance",
    "waveform_length",
    "willison_amplitude",
    "zero_crossings",
]


def _as_window(x: NDArray[np.float64], minimum: int = 1) -> NDArray[np.float64]:
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size < minimum:
        raise ValueError(f"the window needs at least {minimum} samples, got {samples.size}")
    return samples


def mean_absolute_value(x: NDArray[np.float64]) -> float:
    """Mean absolute value, an estimator of the mean amplitude of the window."""
    return float(np.mean(np.abs(_as_window(x))))


def mean_absolute_value_slope(x: NDArray[np.float64], n_segments: int) -> NDArray[np.float64]:
    """Differences between the mean absolute values of adjacent equal length segments.

    Returns ``n_segments - 1`` values. Trailing samples that do not fill a whole
    segment are discarded, so the segments are of exactly equal length and the
    differences are comparable.
    """
    samples = _as_window(x, minimum=2)
    if n_segments < 2:
        raise ValueError("n_segments must be at least 2")
    segment_length = samples.size // n_segments
    if segment_length < 1:
        raise ValueError("the window is too short for the requested number of segments")
    trimmed = samples[: segment_length * n_segments].reshape(n_segments, segment_length)
    values = np.mean(np.abs(trimmed), axis=1)
    return np.asarray(np.diff(values), dtype=np.float64)


def zero_crossings(x: NDArray[np.float64], threshold: float = 0.0) -> int:
    """Number of sign changes whose associated step exceeds ``threshold``.

    The threshold suppresses crossings produced by low amplitude noise wandering about
    zero. With ``threshold = 0`` the count is the plain number of sign changes.
    """
    samples = _as_window(x, minimum=2)
    if threshold < 0.0:
        raise ValueError("threshold must not be negative")
    sign_change = samples[:-1] * samples[1:] < 0.0
    step = np.abs(samples[:-1] - samples[1:]) >= threshold
    return int(np.count_nonzero(sign_change & step))


def slope_sign_changes(x: NDArray[np.float64], threshold: float = 0.0) -> int:
    """Number of interior samples that are a local maximum or minimum.

    A sample ``x[i]`` is counted when ``(x[i] - x[i-1]) (x[i] - x[i+1]) >= threshold``,
    which is positive at a turning point and negative on a monotone run. The threshold
    suppresses turning points produced by noise.
    """
    samples = _as_window(x, minimum=3)
    if threshold < 0.0:
        raise ValueError("threshold must not be negative")
    centre = samples[1:-1]
    product = (centre - samples[:-2]) * (centre - samples[2:])
    if threshold == 0.0:
        return int(np.count_nonzero(product > 0.0))
    return int(np.count_nonzero(product >= threshold))


def waveform_length(x: NDArray[np.float64]) -> float:
    """Cumulative length of the waveform, a joint measure of amplitude and frequency."""
    samples = _as_window(x, minimum=2)
    return float(np.sum(np.abs(np.diff(samples))))


def root_mean_square(x: NDArray[np.float64]) -> float:
    """Root mean square amplitude, which is proportional to muscle force under
    constant force isometric conditions."""
    samples = _as_window(x)
    return float(np.sqrt(np.mean(samples**2)))


def variance(x: NDArray[np.float64]) -> float:
    """Variance about zero, normalised by ``N - 1``."""
    samples = _as_window(x, minimum=2)
    return float(np.sum(samples**2) / (samples.size - 1))


def integrated_emg(x: NDArray[np.float64]) -> float:
    """Sum of absolute values over the window."""
    return float(np.sum(np.abs(_as_window(x))))


def willison_amplitude(x: NDArray[np.float64], threshold: float) -> int:
    """Number of consecutive differences that exceed ``threshold``.

    Willison introduced the count as an index of the number of active motor units. The
    threshold has to be set relative to the noise floor of the recording; a common
    choice is a small multiple of the resting standard deviation.
    """
    samples = _as_window(x, minimum=2)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    return int(np.count_nonzero(np.abs(np.diff(samples)) > threshold))


def autoregressive_coefficients(x: NDArray[np.float64], order: int = 4) -> NDArray[np.float64]:
    """Yule Walker autoregressive coefficients ``a[1] ... a[order]``.

    The sign convention is ``x[n] = sum_k a[k] x[n - k] + e[n]``. As a check with a
    closed form answer, a noiseless sinusoid at angular frequency ``w`` satisfies the
    exact recursion ``x[n] = 2 cos(w) x[n-1] - x[n-2]``, so an order two fit returns
    ``a = (2 cos w, -1)``.
    """
    samples = _as_window(x, minimum=order + 1)
    return yule_walker(samples, order).coefficients


@dataclass(frozen=True, slots=True, eq=False)
class TimeDomainFeatures:
    """The full time domain feature vector for one window."""

    mean_absolute_value: float
    mean_absolute_value_slope: NDArray[np.float64]
    zero_crossings: int
    slope_sign_changes: int
    waveform_length: float
    root_mean_square: float
    variance: float
    integrated_emg: float
    willison_amplitude: int
    autoregressive_coefficients: NDArray[np.float64]

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten the feature set into a single vector, in declaration order."""
        return np.concatenate(
            [
                np.array(
                    [
                        self.mean_absolute_value,
                        float(self.zero_crossings),
                        float(self.slope_sign_changes),
                        self.waveform_length,
                        self.root_mean_square,
                        self.variance,
                        self.integrated_emg,
                        float(self.willison_amplitude),
                    ],
                    dtype=np.float64,
                ),
                np.asarray(self.mean_absolute_value_slope, dtype=np.float64),
                np.asarray(self.autoregressive_coefficients, dtype=np.float64),
            ]
        )


def time_domain_features(
    x: NDArray[np.float64],
    *,
    amplitude_threshold: float,
    slope_threshold: float | None = None,
    n_segments: int = 4,
    ar_order: int = 4,
) -> TimeDomainFeatures:
    """Compute the whole time domain feature set for one window.

    Args:
        x: The window.
        amplitude_threshold: Threshold used by zero crossings and Willison amplitude.
            It has the units of the signal and a common choice is three times the
            standard deviation of a resting segment.
        slope_threshold: Threshold used by slope sign changes. It has the units of the
            signal squared, because the quantity it is compared against is a product of
            two first differences rather than an amplitude. Setting it to the same
            numeric value as ``amplitude_threshold`` would therefore be a units error.
            A defensible choice is the variance of the first difference of a resting
            segment, which is the scale on which noise alone produces turning points.
            Defaults to ``amplitude_threshold ** 2``, which has the right units and the
            right scaling behaviour but is usually far too large to be useful on an
            oversampled record, where consecutive samples differ by much less than the
            signal amplitude.
        n_segments: Number of segments used by the mean absolute value slope.
        ar_order: Order of the autoregressive fit.

    Every threshold scales as its own units require, so multiplying the signal by ``g``
    and both thresholds by ``g`` and ``g ** 2`` respectively leaves all three counting
    features unchanged.
    """
    samples = _as_window(x, minimum=max(3, ar_order + 1))
    slope = amplitude_threshold**2 if slope_threshold is None else slope_threshold
    return TimeDomainFeatures(
        mean_absolute_value=mean_absolute_value(samples),
        mean_absolute_value_slope=mean_absolute_value_slope(samples, n_segments),
        zero_crossings=zero_crossings(samples, amplitude_threshold),
        slope_sign_changes=slope_sign_changes(samples, slope),
        waveform_length=waveform_length(samples),
        root_mean_square=root_mean_square(samples),
        variance=variance(samples),
        integrated_emg=integrated_emg(samples),
        willison_amplitude=willison_amplitude(samples, amplitude_threshold),
        autoregressive_coefficients=autoregressive_coefficients(samples, ar_order),
    )
