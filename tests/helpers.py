"""Measurement helpers and tolerance derivations shared by the test suite.

Every tolerance in this suite is derived from the measurement it constrains, never from
the error that happened to be observed while writing the test. The helpers here name
the three scales that the derivations use.

Sampling interval
    The natural quantum of any timing quantity. A delay estimated from a sampled signal
    cannot be pinned more finely than one sample without an interpolation argument, so
    timing tolerances are expressed in samples.

Frequency resolution
    The natural quantum of any spectral quantity. The Welch estimate resolves
    ``sample_rate / segment_samples``, so spectral tolerances are expressed in bins.

Binomial standard error
    The natural scale of any estimated rate. A proportion estimated from ``n``
    independent trials has standard error at most ``0.5 / sqrt(n)``, so tolerances on
    detection and false positive rates are expressed as multiples of that.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.filters import FilterDesign, apply_causal
from myoelectric.model.sampling import SamplingSpec

SAMPLE_RATE_HZ = 2000.0


def binomial_tolerance(n_trials: int, sigmas: float = 4.0) -> float:
    """Tolerance for a rate estimated from ``n_trials`` independent trials.

    The standard error of a proportion is ``sqrt(p (1 - p) / n)``, which is largest at
    ``p = 0.5`` where it equals ``0.5 / sqrt(n)``. Using that worst case keeps the
    tolerance independent of the value being pinned, which is what makes it a property
    of the experiment rather than of the answer.
    """
    return sigmas * 0.5 / math.sqrt(n_trials)


def floating_point_bound(n_samples: int, unit_scale: float = 1.0) -> float:
    """Bound on the rounding accumulated by an ``n_samples`` long computation.

    A double precision operation introduces a relative error of at most one unit in the
    last place, ``eps = 2.2e-16``. A recursive filter followed by a mean over ``n``
    samples performs on the order of ``n`` dependent operations on values bounded by
    ``unit_scale``, so the accumulated absolute error is bounded by
    ``n * eps * unit_scale``.

    This term exists because a tolerance must never sit on its own boundary. The
    settling bound below is an exact arithmetic argument and for a well settled filter
    it falls below the rounding noise of the very computation it is meant to constrain,
    at which point the comparison tests floating point reproducibility rather than
    filter behaviour and fails on a different machine by a few units in the last place.
    Adding this term keeps the comparison about the filter.
    """
    return n_samples * float(np.finfo(np.float64).eps) * unit_scale


def settling_bound(design: FilterDesign, settle_samples: int, length: int) -> float:
    """Upper bound on the deviation of a causal filter from its steady state response.

    The output of a linear filter at sample ``n`` is ``sum_k h[k] x[n - k]``. Measuring
    the response only from sample ``settle_samples`` onwards discards the contribution
    of the tail of ``h``, so for an input bounded by one in magnitude the difference
    between the measured value and the steady state value is at most the sum of the
    absolute values of the impulse response beyond that point. Computing that sum from
    the design itself gives a tolerance that depends only on the filter, and it is what
    every steady state gain test in this suite uses.

    The rounding bound of :func:`floating_point_bound` is added, so the result is a
    tolerance that a steady state measurement can actually meet on any machine.
    """
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    response = apply_causal(design, impulse)
    tail = float(np.sum(np.abs(response[settle_samples:])))
    return tail + floating_point_bound(length)


def sine(
    frequency_hz: float, sample_rate_hz: float, duration_s: float, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """A sine of the given frequency, sampled from zero phase."""
    times = SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=duration_s).times()
    return np.asarray(amplitude * np.sin(2.0 * np.pi * frequency_hz * times), dtype=np.float64)


def component_amplitude(
    x: NDArray[np.float64], frequency_hz: float, sample_rate_hz: float
) -> float:
    """Amplitude of the component at ``frequency_hz``, by projection onto that frequency.

    Projection is used rather than a peak of a periodogram because it has no window
    leakage and no bin quantisation, so the estimate is exact for a signal that is a sum
    of sinusoids observed over a whole number of periods of each.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    times = np.arange(samples.size, dtype=np.float64) / sample_rate_hz
    angle = 2.0 * np.pi * frequency_hz * times
    real = 2.0 * float(np.mean(samples * np.cos(angle)))
    imaginary = 2.0 * float(np.mean(samples * np.sin(angle)))
    return math.hypot(real, imaginary)


def energy_centroid(x: NDArray[np.float64]) -> float:
    """Centre of mass of the squared signal, in samples.

    This locates a burst without needing to find a peak, so it is insensitive to the
    sample grid and can resolve a shift to a fraction of a sample.
    """
    weight = np.asarray(x, dtype=np.float64) ** 2
    total = float(np.sum(weight))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(np.arange(weight.size, dtype=np.float64) * weight) / total)
