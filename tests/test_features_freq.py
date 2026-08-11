"""Property tests for the frequency domain feature library.

Tolerances on spectral quantities are expressed in frequency bins, because the bin
width is what a Welch estimate can resolve. Asserting a spectral feature to better than
one bin would be asserting more than the estimator supports.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoelectric.algorithm.features_freq import (
    PowerSpectrum,
    frequency_domain_features,
    mean_frequency,
    median_frequency,
    spectral_moment,
    welch_spectrum,
)

# Three sinusoids whose powers are 1, 3 and 1 in the ratio, placed symmetrically about
# 150 Hz. The mean frequency is therefore exactly
# (50 * 1 + 150 * 3 + 250 * 1) / 5 = 150 Hz, and half of the total power is reached
# strictly inside the middle component, so the median frequency is 150 Hz as well.
COMPONENT_HZ = (50.0, 150.0, 250.0)
COMPONENT_AMPLITUDE = (1.0, float(np.sqrt(3.0)), 1.0)
EXPECTED_HZ = 150.0

SEGMENT_S = 0.5
DURATION_S = 4.0


def _analytic_signal(sample_rate_hz: float, gain: float = 1.0) -> np.ndarray:
    index = np.arange(int(DURATION_S * sample_rate_hz), dtype=np.float64)
    total = np.zeros(index.size, dtype=np.float64)
    for frequency, amplitude in zip(COMPONENT_HZ, COMPONENT_AMPLITUDE, strict=True):
        total += amplitude * np.sin(2.0 * np.pi * frequency * index / sample_rate_hz)
    return gain * total


def test_spectrum_resolution_is_the_reciprocal_of_the_segment_length(
    sample_rate_hz: float,
) -> None:
    """The Welch estimate resolves one over the segment length."""
    spectrum = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    assert spectrum.resolution_hz == pytest.approx(1.0 / SEGMENT_S, rel=1e-12)
    assert float(spectrum.frequencies_hz[1] - spectrum.frequencies_hz[0]) == pytest.approx(
        spectrum.resolution_hz, rel=1e-12
    )


def test_median_and_mean_frequency_match_the_analytic_value(sample_rate_hz: float) -> None:
    """A signal with a known line spectrum has known median and mean frequencies.

    Tolerance: one frequency bin, which is the resolution of the estimate. Every
    component sits exactly on a bin centre, so window leakage is symmetric about each
    component and cancels in both moments.
    """
    spectrum = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    tolerance = spectrum.resolution_hz
    assert median_frequency(spectrum) == pytest.approx(EXPECTED_HZ, abs=tolerance)
    assert mean_frequency(spectrum) == pytest.approx(EXPECTED_HZ, abs=tolerance)


def test_median_frequency_is_interpolated_between_bins() -> None:
    """The half power point is placed between bins rather than snapped to one.

    With powers 1 and 3 at 0 Hz and 10 Hz, half of the total of 4 is reached one third
    of the way through the second bin, so the answer is 10 / 3 Hz. Snapping to the bin
    grid would return 10 Hz.
    """
    spectrum = PowerSpectrum(
        frequencies_hz=np.array([0.0, 10.0]),
        power=np.array([1.0, 3.0]),
        resolution_hz=10.0,
        method="hand constructed",
    )
    assert median_frequency(spectrum) == pytest.approx(10.0 / 3.0, rel=1e-12)


def test_spectral_moments_follow_their_definitions(sample_rate_hz: float) -> None:
    """The zeroth moment is total power and the first over the zeroth is mean frequency."""
    spectrum = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    moment_0 = spectral_moment(spectrum, 0)
    moment_1 = spectral_moment(spectrum, 1)
    assert moment_1 / moment_0 == pytest.approx(mean_frequency(spectrum), rel=1e-12)
    assert spectrum.total_power == pytest.approx(moment_0, rel=1e-12)


def test_total_power_matches_the_mean_square_of_the_signal(sample_rate_hz: float) -> None:
    """Parseval: integrating the density over frequency returns the mean square.

    Tolerance: one per cent relative. The Welch estimate windows each segment, so the
    equality holds up to the leakage that the window spreads outside the analysed band
    and up to the segments discarded at the end of the record.
    """
    signal = _analytic_signal(sample_rate_hz)
    spectrum = welch_spectrum(signal, sample_rate_hz, segment_s=SEGMENT_S)
    assert spectrum.total_power == pytest.approx(float(np.mean(signal**2)), rel=0.01)


@pytest.mark.parametrize("gain", [0.5, 3.0])
def test_frequency_features_do_not_scale_and_power_scales_quadratically(
    gain: float, sample_rate_hz: float
) -> None:
    """Median and mean frequency are unchanged by a gain; total power scales by its square."""
    base = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    scaled = welch_spectrum(
        _analytic_signal(sample_rate_hz, gain), sample_rate_hz, segment_s=SEGMENT_S
    )
    assert median_frequency(scaled) == pytest.approx(median_frequency(base), rel=1e-12)
    assert mean_frequency(scaled) == pytest.approx(mean_frequency(base), rel=1e-12)
    assert scaled.total_power == pytest.approx(gain**2 * base.total_power, rel=1e-12)


def test_band_restriction_keeps_only_the_requested_bins(sample_rate_hz: float) -> None:
    """Restricting the band drops out of band power and records the restriction."""
    spectrum = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    restricted = spectrum.band(100.0, 200.0)
    assert float(restricted.frequencies_hz.min()) >= 100.0
    assert float(restricted.frequencies_hz.max()) <= 200.0
    assert restricted.resolution_hz == spectrum.resolution_hz
    assert "restricted" in restricted.method
    assert median_frequency(restricted) == pytest.approx(EXPECTED_HZ, abs=restricted.resolution_hz)


def test_frequency_feature_set_reports_its_method(sample_rate_hz: float) -> None:
    """The feature set carries the estimation method and resolution with the numbers."""
    spectrum = welch_spectrum(_analytic_signal(sample_rate_hz), sample_rate_hz, segment_s=SEGMENT_S)
    features = frequency_domain_features(spectrum)
    assert "Welch" in features.method
    assert features.resolution_hz == spectrum.resolution_hz
    assert features.root_mean_square_frequency_hz == pytest.approx(
        float(np.sqrt(features.spectral_moment_2 / features.spectral_moment_0)), rel=1e-12
    )
    assert features.mean_frequency_hz <= features.root_mean_square_frequency_hz


def test_spectral_functions_reject_degenerate_input() -> None:
    """A spectrum with no power has no median or mean frequency."""
    empty = PowerSpectrum(
        frequencies_hz=np.array([0.0, 1.0]),
        power=np.array([0.0, 0.0]),
        resolution_hz=1.0,
        method="hand constructed",
    )
    with pytest.raises(ValueError):
        median_frequency(empty)
    with pytest.raises(ValueError):
        mean_frequency(empty)
    with pytest.raises(ValueError):
        spectral_moment(empty, -1)
    with pytest.raises(ValueError):
        empty.band(10.0, 20.0)
