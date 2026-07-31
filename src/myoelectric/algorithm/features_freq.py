"""Frequency domain feature library for myoelectric signals.

Spectral estimation method. The power spectral density is estimated by the averaged
periodogram of Welch (1967): the record is split into overlapping segments, each
segment is multiplied by a Hann window, the squared magnitude of its discrete Fourier
transform is formed, and the results are averaged. Averaging ``k`` segments reduces the
variance of the estimate by roughly a factor of ``k`` at the cost of a frequency
resolution of ``sample_rate / segment_length``. Every spectral feature below is
computed from that estimate, and every tolerance placed on a spectral feature in the
test suite is expressed as a multiple of that resolution.

Features, for a spectrum ``P[j]`` at frequencies ``f[j]`` with bin width ``df``:

Mean frequency
    ``MNF = sum(f[j] P[j]) / sum(P[j])``, the first spectral moment normalised by the
    zeroth. Stulen and De Luca (1981) analysed both mean and median frequency as
    estimators of muscle fibre conduction velocity.

Median frequency
    ``MDF`` is the frequency that divides the spectrum into two halves of equal power.
    It is obtained here by linear interpolation of the cumulative power between the two
    bins that bracket half of the total, which gives an estimate finer than one bin
    rather than quantising the answer to the bin grid.

Spectral moments
    ``SM[k] = sum(f[j]^k P[j]) df``. The zeroth moment is total power, the first
    divided by the zeroth is the mean frequency, and the ratio ``SM[2] / SM[0]`` gives
    the mean square frequency, whose square root is used in fatigue indices.

Median frequency is preferred over mean frequency as a fatigue index because it is less
sensitive to additive noise outside the signal band, since noise contributes to the
numerator of the mean in proportion to its frequency but only shifts the median once it
displaces half the cumulative power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

__all__ = [
    "FrequencyDomainFeatures",
    "PowerSpectrum",
    "frequency_domain_features",
    "mean_frequency",
    "median_frequency",
    "spectral_moment",
    "welch_spectrum",
]


@dataclass(frozen=True, slots=True, eq=False)
class PowerSpectrum:
    """A one sided power spectral density estimate on a uniform frequency grid."""

    frequencies_hz: NDArray[np.float64]
    power: NDArray[np.float64]
    resolution_hz: float
    method: str

    def __post_init__(self) -> None:
        if self.frequencies_hz.shape != self.power.shape:
            raise ValueError("frequencies_hz and power must have the same shape")
        if self.frequencies_hz.size < 2:
            raise ValueError("a spectrum needs at least two bins")

    def band(self, low_hz: float, high_hz: float) -> PowerSpectrum:
        """Restrict the spectrum to ``[low_hz, high_hz]``.

        Restricting to the signal band before computing a fatigue index is standard
        practice, because out of band noise otherwise contributes to the moments.
        """
        mask = (self.frequencies_hz >= low_hz) & (self.frequencies_hz <= high_hz)
        if int(np.count_nonzero(mask)) < 2:
            raise ValueError(f"the band {low_hz} to {high_hz} Hz contains fewer than two bins")
        return PowerSpectrum(
            frequencies_hz=self.frequencies_hz[mask],
            power=self.power[mask],
            resolution_hz=self.resolution_hz,
            method=f"{self.method}, restricted to {low_hz:g}-{high_hz:g} Hz",
        )

    @property
    def total_power(self) -> float:
        """Total power, the zeroth spectral moment."""
        return spectral_moment(self, 0)


def welch_spectrum(
    x: NDArray[np.float64],
    sample_rate_hz: float,
    segment_s: float = 0.25,
    overlap: float = 0.5,
    window: str = "hann",
) -> PowerSpectrum:
    """Welch power spectral density estimate.

    Args:
        x: The record.
        sample_rate_hz: Sample rate.
        segment_s: Segment length in seconds. The frequency resolution of the estimate
            is ``1 / segment_s``, so a 0.25 s segment resolves 4 Hz.
        overlap: Fractional overlap between consecutive segments, in ``[0, 1)``.
        window: Any window name accepted by ``scipy.signal.get_window``.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1)")
    nperseg = min(samples.size, max(16, round(float(segment_s) * float(sample_rate_hz))))
    noverlap = round(float(overlap) * nperseg)
    frequencies, power = sp_signal.welch(
        samples,
        fs=sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    resolution = sample_rate_hz / nperseg
    return PowerSpectrum(
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        power=np.asarray(power, dtype=np.float64),
        resolution_hz=resolution,
        method=(
            f"Welch averaged periodogram, {window} window, {nperseg} sample segments, "
            f"{overlap:.0%} overlap, resolution {resolution:.3g} Hz"
        ),
    )


def spectral_moment(spectrum: PowerSpectrum, order: int) -> float:
    """Spectral moment of the given order, ``sum(f^order P) df``."""
    if order < 0:
        raise ValueError("order must not be negative")
    weights = spectrum.frequencies_hz**order
    return float(np.sum(weights * spectrum.power) * spectrum.resolution_hz)


def mean_frequency(spectrum: PowerSpectrum) -> float:
    """Power weighted mean frequency in hertz."""
    total = float(np.sum(spectrum.power))
    if total <= 0.0:
        raise ValueError("cannot compute a mean frequency of a spectrum with no power")
    return float(np.sum(spectrum.frequencies_hz * spectrum.power) / total)


def median_frequency(spectrum: PowerSpectrum) -> float:
    """Frequency that splits the spectrum into two halves of equal power.

    The cumulative power is formed on the bin grid and the half power point is located
    by linear interpolation between the bracketing bins. Interpolation is used rather
    than returning the first bin at or above half power, because the latter quantises
    the estimate to the bin width and introduces a systematic upward bias of half a
    bin.
    """
    power = spectrum.power
    total = float(np.sum(power))
    if total <= 0.0:
        raise ValueError("cannot compute a median frequency of a spectrum with no power")
    cumulative = np.cumsum(power)
    half = 0.5 * total
    index = int(np.searchsorted(cumulative, half, side="left"))
    if index == 0:
        return float(spectrum.frequencies_hz[0])
    if index >= power.size:
        return float(spectrum.frequencies_hz[-1])
    lower_cumulative = float(cumulative[index - 1])
    upper_cumulative = float(cumulative[index])
    span = upper_cumulative - lower_cumulative
    fraction = 0.0 if span <= 0.0 else (half - lower_cumulative) / span
    lower_frequency = float(spectrum.frequencies_hz[index - 1])
    upper_frequency = float(spectrum.frequencies_hz[index])
    return lower_frequency + fraction * (upper_frequency - lower_frequency)


@dataclass(frozen=True, slots=True)
class FrequencyDomainFeatures:
    """The frequency domain feature set for one window."""

    median_frequency_hz: float
    mean_frequency_hz: float
    spectral_moment_0: float
    spectral_moment_1: float
    spectral_moment_2: float
    root_mean_square_frequency_hz: float
    resolution_hz: float
    method: str


def frequency_domain_features(spectrum: PowerSpectrum) -> FrequencyDomainFeatures:
    """Compute every frequency domain feature from one spectrum estimate."""
    moment_0 = spectral_moment(spectrum, 0)
    moment_1 = spectral_moment(spectrum, 1)
    moment_2 = spectral_moment(spectrum, 2)
    rms_frequency = float(np.sqrt(moment_2 / moment_0)) if moment_0 > 0.0 else 0.0
    return FrequencyDomainFeatures(
        median_frequency_hz=median_frequency(spectrum),
        mean_frequency_hz=mean_frequency(spectrum),
        spectral_moment_0=moment_0,
        spectral_moment_1=moment_1,
        spectral_moment_2=moment_2,
        root_mean_square_frequency_hz=rms_frequency,
        resolution_hz=spectrum.resolution_hz,
        method=spectrum.method,
    )
