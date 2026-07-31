"""Autoregressive modelling by the Yule Walker equations.

Two parts of this library need the same machinery. The time domain feature set
includes autoregressive coefficients, which Graupe and Cline (1975) introduced as a
compact description of the myoelectric waveform for prosthesis control. The Bonato
style onset detector needs to whiten the signal before applying a statistical test,
and whitening is the residual of the same autoregressive model.

The coefficients are obtained from the biased autocorrelation estimate by the Levinson
Durbin recursion. The recursion is a direct algorithm, not an iterative solve: it
performs a fixed number of arithmetic operations determined only by the model order,
so its result is reproducible on any machine to within floating point rounding. The
biased autocorrelation estimate is used because it is guaranteed to be positive
semidefinite, which in turn guarantees that the recursion returns a stable model.

Sign convention used throughout::

    x[n] = sum_{k=1..p} a[k] x[n - k] + e[n]

so the whitening filter is ``e[n] = x[n] - sum_k a[k] x[n - k]`` and the returned
coefficient vector holds ``a[1] ... a[p]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ARModel", "autocorrelation", "levinson_durbin", "whiten", "yule_walker"]


@dataclass(frozen=True, slots=True, eq=False)
class ARModel:
    """An autoregressive model of a stationary signal."""

    coefficients: NDArray[np.float64]
    noise_variance: float

    @property
    def order(self) -> int:
        """Model order."""
        return int(self.coefficients.size)


def autocorrelation(x: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Biased autocorrelation estimate for lags ``0 ... max_lag``.

    The estimate is ``r[k] = (1 / n) sum_{i=0}^{n-1-k} x[i] x[i + k]``. Dividing by
    ``n`` rather than by ``n - k`` tapers the estimate towards zero at long lags, which
    costs a relative bias of ``k / n`` but keeps the sequence positive semidefinite.

    Only the lags that are asked for are computed, which costs ``max_lag`` inner
    products of length ``n``. Forming the full correlation and discarding all but the
    first few lags would cost ``n`` times as much and is the usual way this function is
    written.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if max_lag < 0:
        raise ValueError("max_lag must not be negative")
    if samples.size <= max_lag:
        raise ValueError(
            f"need more than {max_lag} samples to estimate lags up to {max_lag}, "
            f"got {samples.size}"
        )
    n = samples.size
    lags = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        lags[lag] = float(np.dot(samples[: n - lag], samples[lag:]))
    return np.asarray(lags / n, dtype=np.float64)


def levinson_durbin(r: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Solve the Yule Walker system for the autocorrelation sequence ``r``.

    Args:
        r: Autocorrelation values for lags ``0 ... p``.

    Returns:
        The coefficient vector ``a[1] ... a[p]`` and the prediction error variance.
    """
    values = np.asarray(r, dtype=np.float64).ravel()
    order = values.size - 1
    if order < 1:
        raise ValueError("the autocorrelation sequence must cover at least lag 1")
    if values[0] <= 0.0:
        raise ValueError("the zero lag autocorrelation must be positive")

    coefficients = np.zeros(order, dtype=np.float64)
    error = float(values[0])
    for k in range(order):
        acc = values[k + 1] - float(np.dot(coefficients[:k], values[k:0:-1]))
        reflection = acc / error if error > 0.0 else 0.0
        previous = coefficients[:k].copy()
        coefficients[k] = reflection
        if k > 0:
            coefficients[:k] = previous - reflection * previous[::-1]
        error *= 1.0 - reflection * reflection
        if error <= 0.0:
            error = 0.0
            break
    return coefficients, error


def yule_walker(x: NDArray[np.float64], order: int) -> ARModel:
    """Fit an autoregressive model of the given order to ``x``."""
    if order < 1:
        raise ValueError("order must be at least 1")
    r = autocorrelation(x, order)
    if r[0] <= 0.0:
        raise ValueError("cannot fit an autoregressive model to a constant zero signal")
    coefficients, error = levinson_durbin(r)
    return ARModel(coefficients=coefficients, noise_variance=error)


def whiten(x: NDArray[np.float64], model: ARModel) -> NDArray[np.float64]:
    """Apply the whitening filter of ``model`` to ``x``.

    The first ``order`` samples of the result are set to zero because the filter has no
    history there; callers that use the residual for a statistical test should discard
    that leading region.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    order = model.order
    residual = samples.copy()
    for lag in range(1, order + 1):
        residual[order:] -= model.coefficients[lag - 1] * samples[order - lag : samples.size - lag]
    residual[:order] = 0.0
    return residual
