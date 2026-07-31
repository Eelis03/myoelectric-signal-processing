"""Property tests for the time domain feature library.

Every feature is checked on a signal whose answer is known by inspection or in closed
form, so a failure identifies a wrong implementation rather than a changed number.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from myoelectric.algorithm.autoregressive import levinson_durbin, whiten, yule_walker
from myoelectric.algorithm.features_time import (
    autoregressive_coefficients,
    integrated_emg,
    mean_absolute_value,
    mean_absolute_value_slope,
    root_mean_square,
    slope_sign_changes,
    time_domain_features,
    variance,
    waveform_length,
    willison_amplitude,
    zero_crossings,
)
from tests.helpers import floating_point_bound

# A triangle that rises from 0 to 10 in unit steps and falls back, so its waveform
# length is 20 by inspection, it has exactly one turning point, and it never crosses
# zero.
_RAMP = np.arange(0.0, 11.0)
TRIANGLE = np.concatenate([_RAMP, _RAMP[-2::-1]])

# Eight periods of a square wave with 25 samples at each level, so the sign changes at
# every level boundary. There are sixteen level blocks and therefore fifteen boundaries.
_HALF_PERIOD = 25
_PERIODS = 8
SQUARE = np.tile(
    np.concatenate([np.full(_HALF_PERIOD, 1.0), np.full(_HALF_PERIOD, -1.0)]), _PERIODS
)


def test_zero_crossings_on_a_square_wave() -> None:
    """A square wave with sixteen level blocks has fifteen sign changes."""
    assert SQUARE.size == 2 * _HALF_PERIOD * _PERIODS
    assert zero_crossings(SQUARE) == 2 * _PERIODS - 1


def test_zero_crossings_threshold_suppresses_small_crossings() -> None:
    """A crossing whose step is below the threshold is not counted."""
    small = 0.01 * SQUARE
    assert zero_crossings(small, threshold=0.0) == 2 * _PERIODS - 1
    assert zero_crossings(small, threshold=0.1) == 0


def test_zero_crossings_ignores_a_signal_that_touches_zero_without_crossing() -> None:
    """Touching zero is not a crossing, because the product of the two samples is zero."""
    touching = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert zero_crossings(touching) == 0


def test_waveform_length_on_a_triangle_wave() -> None:
    """The triangle rises by ten and falls by ten, so its waveform length is twenty."""
    assert waveform_length(TRIANGLE) == pytest.approx(20.0, abs=floating_point_bound(TRIANGLE.size))


def test_slope_sign_changes_on_a_triangle_wave() -> None:
    """A single peaked triangle has exactly one turning point."""
    assert slope_sign_changes(TRIANGLE) == 1


def test_slope_sign_changes_on_a_zigzag() -> None:
    """A zigzag alternating every sample has a turning point at every interior sample."""
    zigzag = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert slope_sign_changes(zigzag) == zigzag.size - 2


def test_slope_sign_changes_threshold_has_squared_units() -> None:
    """The threshold is compared against a product of two differences, not an amplitude."""
    zigzag = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    # Each interior sample gives a product of magnitude 1 * 1 = 1.
    assert slope_sign_changes(zigzag, threshold=0.5) == 3
    assert slope_sign_changes(zigzag, threshold=2.0) == 0


def test_willison_amplitude_counts_differences_above_the_threshold() -> None:
    """Constructed steps of 3, 1, 4 and 1 give two counts above a threshold of 2."""
    signal = np.array([0.0, 3.0, 4.0, 8.0, 9.0])
    assert willison_amplitude(signal, threshold=2.0) == 2
    assert willison_amplitude(signal, threshold=0.5) == 4
    assert willison_amplitude(signal, threshold=5.0) == 0


def test_amplitude_features_of_a_sine_match_their_closed_forms(sample_rate_hz: float) -> None:
    """Root mean square is exactly ``A / sqrt(2)`` and mean absolute value is ``2A / pi``.

    Tolerances. The sum of squares of a sine over a whole number of periods is exactly
    ``n A^2 / 2`` for any sample count above two per period, so root mean square is
    checked against the accumulated rounding bound alone. The mean of the absolute value
    is a Riemann sum for the integral of a function of total variation four over one
    period, so its error is bounded by ``4 / (2 m)`` for ``m`` samples per period.
    """
    amplitude = 3.0
    samples_per_period = 1000
    periods = 4
    index = np.arange(samples_per_period * periods, dtype=np.float64)
    signal = amplitude * np.sin(2.0 * np.pi * index / samples_per_period)

    assert root_mean_square(signal) == pytest.approx(
        amplitude / np.sqrt(2.0), abs=floating_point_bound(signal.size, amplitude)
    )
    assert mean_absolute_value(signal) == pytest.approx(
        2.0 * amplitude / np.pi, abs=4.0 * amplitude / (2.0 * samples_per_period)
    )
    assert integrated_emg(signal) == pytest.approx(
        signal.size * mean_absolute_value(signal),
        rel=1e-12,
    )


def test_variance_is_normalised_by_one_less_than_the_sample_count() -> None:
    """Variance is the sum of squares divided by ``N - 1``, with no mean removed."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    assert variance(signal) == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 3.0, rel=1e-12)


def test_mean_absolute_value_slope_differences_adjacent_segments() -> None:
    """Four segments of known amplitude give three known differences."""
    signal = np.concatenate(
        [np.full(10, 1.0), np.full(10, -2.0), np.full(10, 4.0), np.full(10, -1.0)]
    )
    slope = mean_absolute_value_slope(signal, n_segments=4)
    assert slope.size == 3
    assert slope == pytest.approx(np.array([1.0, 2.0, -3.0]), rel=1e-12)


def test_autoregressive_coefficients_of_a_sinusoid_are_analytic(sample_rate_hz: float) -> None:
    """A noiseless sinusoid satisfies ``x[n] = 2 cos(w) x[n-1] - x[n-2]`` exactly.

    Tolerance. The biased autocorrelation estimator tapers lag ``k`` by ``1 - k / n``,
    so for an order ``p`` fit over ``n`` samples the relative error of the estimate is
    on the order of ``p / n``. Ten times that is used, which is generous against the
    estimator bias and still far tighter than any accidental agreement.
    """
    frequency_hz = 120.0
    n_samples = 4096
    index = np.arange(n_samples, dtype=np.float64)
    omega = 2.0 * np.pi * frequency_hz / sample_rate_hz
    signal = np.sin(omega * index)

    coefficients = autoregressive_coefficients(signal, order=2)
    tolerance = 10.0 * 2.0 / n_samples
    assert coefficients[0] == pytest.approx(2.0 * np.cos(omega), abs=tolerance)
    assert coefficients[1] == pytest.approx(-1.0, abs=tolerance)


def test_autoregressive_coefficients_are_scale_invariant(sample_rate_hz: float) -> None:
    """Scaling the signal does not change the correlation structure it describes."""
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(2048)
    base = autoregressive_coefficients(signal, order=4)
    scaled = autoregressive_coefficients(7.5 * signal, order=4)
    assert scaled == pytest.approx(base, rel=1e-9)


def test_levinson_durbin_reproduces_a_known_autoregressive_process() -> None:
    """The recursion recovers the coefficients of the process that generated the data.

    An order two process with a known pair of coefficients is driven by white noise, and
    the fit is compared against the truth. Tolerance: three standard errors of the
    coefficient estimate, which for an order ``p`` fit over ``n`` samples is
    ``sqrt(p / n)`` per coefficient, that is the sampling error of the estimate rather
    than an observed discrepancy.
    """
    rng = np.random.default_rng(3)
    n_samples = 200_000
    true = np.array([0.6, -0.3])
    noise = rng.standard_normal(n_samples)
    # x[n] = a1 x[n-1] + a2 x[n-2] + e[n] is the all pole filter with denominator
    # [1, -a1, -a2] driven by the noise.
    signal = np.asarray(
        sp_signal.lfilter([1.0], [1.0, -true[0], -true[1]], noise), dtype=np.float64
    )

    model = yule_walker(signal, order=2)
    tolerance = 3.0 * np.sqrt(2.0 / n_samples)
    assert model.coefficients == pytest.approx(true, abs=tolerance)
    assert model.noise_variance == pytest.approx(1.0, abs=0.05)

    residual = whiten(signal, model)
    assert float(np.var(residual[2:])) == pytest.approx(1.0, abs=0.05)


def test_levinson_durbin_rejects_a_degenerate_sequence() -> None:
    """A sequence with no power cannot define an autoregressive model."""
    with pytest.raises(ValueError):
        levinson_durbin(np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("gain", [0.25, 4.0])
def test_features_behave_correctly_under_a_change_of_scale(
    gain: float, sample_rate_hz: float
) -> None:
    """Amplitude features scale, variance scales quadratically, counts do not change."""
    rng = np.random.default_rng(5)
    signal = rng.standard_normal(1024)
    amplitude_threshold = 0.5
    slope_threshold = 0.25

    base = time_domain_features(
        signal, amplitude_threshold=amplitude_threshold, slope_threshold=slope_threshold
    )
    scaled = time_domain_features(
        gain * signal,
        amplitude_threshold=gain * amplitude_threshold,
        slope_threshold=gain**2 * slope_threshold,
    )

    assert scaled.mean_absolute_value == pytest.approx(gain * base.mean_absolute_value, rel=1e-12)
    assert scaled.root_mean_square == pytest.approx(gain * base.root_mean_square, rel=1e-12)
    assert scaled.waveform_length == pytest.approx(gain * base.waveform_length, rel=1e-12)
    assert scaled.integrated_emg == pytest.approx(gain * base.integrated_emg, rel=1e-12)
    assert scaled.variance == pytest.approx(gain**2 * base.variance, rel=1e-12)
    assert scaled.mean_absolute_value_slope == pytest.approx(
        gain * base.mean_absolute_value_slope, rel=1e-12
    )

    assert scaled.zero_crossings == base.zero_crossings
    assert scaled.slope_sign_changes == base.slope_sign_changes
    assert scaled.willison_amplitude == base.willison_amplitude
    assert scaled.autoregressive_coefficients == pytest.approx(
        base.autoregressive_coefficients, rel=1e-9
    )


def test_feature_vector_is_flattened_in_declaration_order() -> None:
    """The flattened vector has the length that the feature set implies."""
    rng = np.random.default_rng(7)
    features = time_domain_features(
        rng.standard_normal(512), amplitude_threshold=0.5, n_segments=4, ar_order=4
    )
    assert features.as_vector().size == 8 + 3 + 4


@pytest.mark.parametrize(
    "call",
    [
        lambda: mean_absolute_value(np.array([])),
        lambda: waveform_length(np.array([1.0])),
        lambda: slope_sign_changes(np.array([1.0, 2.0])),
        lambda: willison_amplitude(np.array([1.0, 2.0]), threshold=0.0),
        lambda: zero_crossings(np.array([1.0, 2.0]), threshold=-1.0),
        lambda: mean_absolute_value_slope(np.array([1.0, 2.0]), n_segments=1),
    ],
)
def test_features_reject_invalid_input(call) -> None:
    """Too short a window or an invalid threshold raises rather than returning nonsense."""
    with pytest.raises(ValueError):
        call()
