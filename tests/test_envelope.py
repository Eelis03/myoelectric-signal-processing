"""Property tests for amplitude estimation and the latency it imposes.

The delay figures quoted for each estimator are design figures. These tests measure the
delay from a step response and compare it against the design, so the two cannot drift
apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoelectric.algorithm.envelope import (
    EnvelopeEstimator,
    ExponentialEnvelope,
    LowPassEnvelope,
    MovingAverageEnvelope,
    MovingRmsEnvelope,
    measure_latency,
)

SAMPLE_RATE_HZ = 2000.0
DURATION_S = 4.0
STEP_S = 2.0


def _carrier_step(sample_rate_hz: float) -> np.ndarray:
    """A constant amplitude carrier that steps from silence to full amplitude.

    A deterministic square carrier is used instead of a noisy record so that the
    envelope of the input is exactly rectangular and the measured latency belongs
    entirely to the estimator. The carrier alternates every sample so that its
    rectified value is constant, which removes any ripple from the input itself. The
    level before the step is zero so that each estimator's step response has a closed
    form: the moving average ramps linearly, the moving root mean square is the square
    root of that ramp, and the exponential estimator is a single exponential.
    """
    n_samples = int(DURATION_S * sample_rate_hz)
    step_index = int(STEP_S * sample_rate_hz)
    carrier = np.where(np.arange(n_samples) % 2 == 0, 1.0, -1.0)
    amplitude = np.where(np.arange(n_samples) < step_index, 0.0, 1.0)
    return np.asarray(carrier * amplitude, dtype=np.float64)


@pytest.mark.parametrize("window_s", [0.050, 0.100, 0.200])
def test_moving_average_nominal_delay_is_half_the_window(window_s: float) -> None:
    """A rectangular kernel of ``n`` samples has a group delay of exactly ``(n - 1) / 2``."""
    estimator = MovingAverageEnvelope(window_s)
    window = estimator.window_samples(SAMPLE_RATE_HZ)
    assert estimator.nominal_delay_samples(SAMPLE_RATE_HZ) == pytest.approx(
        0.5 * (window - 1), rel=1e-12
    )


@pytest.mark.parametrize("time_constant_s", [0.010, 0.025, 0.050])
def test_exponential_nominal_delay_matches_its_closed_form(time_constant_s: float) -> None:
    """A single pole average with coefficient ``a`` has delay ``(1 - a) / a`` at zero frequency."""
    estimator = ExponentialEnvelope(time_constant_s)
    a = estimator.alpha(SAMPLE_RATE_HZ)
    assert estimator.nominal_delay_samples(SAMPLE_RATE_HZ) == pytest.approx(
        (1.0 - a) / a, rel=1e-12
    )


@pytest.mark.parametrize("window_s", [0.050, 0.100, 0.200])
def test_moving_average_measured_latency_matches_its_nominal_delay(window_s: float) -> None:
    """The step response reaches half amplitude at the nominal group delay.

    For a rectangular kernel the output ramps linearly across the window, so the half
    amplitude point falls exactly at the middle of the window, which is the group delay.

    Tolerance: one sample, the quantum of a discrete time delay measurement.
    """
    estimator = MovingAverageEnvelope(window_s)
    signal = _carrier_step(SAMPLE_RATE_HZ)
    step_index = int(STEP_S * SAMPLE_RATE_HZ)
    plateau = slice(step_index + int(0.5 * SAMPLE_RATE_HZ), signal.size)
    measurement = measure_latency(estimator, signal, SAMPLE_RATE_HZ, step_index, plateau)

    one_sample_ms = 1e3 / SAMPLE_RATE_HZ
    assert measurement.latency_ms == pytest.approx(
        measurement.nominal_delay_ms, abs=one_sample_ms
    )


def test_moving_rms_reaches_half_amplitude_in_a_quarter_of_its_window() -> None:
    """The mean square ramps linearly, so its square root reaches half at a quarter window.

    Tolerance: one sample. This is a closed form property of the estimator and it is why
    the moving root mean square appears faster than the moving average of the same
    window without being any less smooth in the steady state.
    """
    window_s = 0.100
    estimator = MovingRmsEnvelope(window_s)
    signal = _carrier_step(SAMPLE_RATE_HZ)
    step_index = int(STEP_S * SAMPLE_RATE_HZ)
    plateau = slice(step_index + int(0.5 * SAMPLE_RATE_HZ), signal.size)
    measurement = measure_latency(estimator, signal, SAMPLE_RATE_HZ, step_index, plateau)

    one_sample_ms = 1e3 / SAMPLE_RATE_HZ
    assert measurement.latency_ms == pytest.approx(
        0.25 * 1e3 * window_s, abs=2.0 * one_sample_ms
    )


def test_exponential_reaches_half_amplitude_at_log_two_of_its_time_constant() -> None:
    """A first order step response crosses half at ``ln 2`` times the time constant.

    Tolerance: one sample. Reporting the nominal group delay as the latency of an
    exponential estimator would overstate it by a factor of ``1 / ln 2``, which is why
    the measured value is the one reported.
    """
    time_constant_s = 0.050
    estimator = ExponentialEnvelope(time_constant_s)
    signal = _carrier_step(SAMPLE_RATE_HZ)
    step_index = int(STEP_S * SAMPLE_RATE_HZ)
    plateau = slice(step_index + int(0.5 * SAMPLE_RATE_HZ), signal.size)
    measurement = measure_latency(estimator, signal, SAMPLE_RATE_HZ, step_index, plateau)

    one_sample_ms = 1e3 / SAMPLE_RATE_HZ
    assert measurement.latency_ms == pytest.approx(
        1e3 * time_constant_s * np.log(2.0), abs=2.0 * one_sample_ms
    )


def test_more_smoothing_costs_more_latency_and_buys_less_ripple() -> None:
    """Across a family of window lengths, latency rises and plateau ripple falls.

    This is the trade off that the amplitude estimator study reports, asserted here as a
    monotone property rather than as a set of numbers.
    """
    rng = np.random.default_rng(909)
    n_samples = int(DURATION_S * SAMPLE_RATE_HZ)
    step_index = int(STEP_S * SAMPLE_RATE_HZ)
    amplitude = np.where(np.arange(n_samples) < step_index, 0.2, 1.0)
    signal = amplitude * rng.standard_normal(n_samples)
    plateau = slice(step_index + int(0.5 * SAMPLE_RATE_HZ), n_samples)

    latencies: list[float] = []
    ripples: list[float] = []
    for window_s in (0.025, 0.050, 0.100, 0.200):
        measurement = measure_latency(
            MovingAverageEnvelope(window_s), signal, SAMPLE_RATE_HZ, step_index, plateau
        )
        latencies.append(measurement.latency_ms)
        ripples.append(measurement.plateau_ripple_percent)

    assert latencies == sorted(latencies)
    assert ripples == sorted(ripples, reverse=True)


@pytest.mark.parametrize(
    "estimator",
    [
        MovingAverageEnvelope(0.1),
        MovingRmsEnvelope(0.1),
        LowPassEnvelope(4.0),
        ExponentialEnvelope(0.05),
    ],
    ids=lambda e: e.name,
)
def test_estimators_satisfy_the_protocol_and_are_non_negative(
    estimator: EnvelopeEstimator,
) -> None:
    """Every estimator is usable through the common protocol and returns amplitudes."""
    assert isinstance(estimator, EnvelopeEstimator)
    signal = _carrier_step(SAMPLE_RATE_HZ)
    envelope = estimator.estimate(signal, SAMPLE_RATE_HZ)
    assert envelope.shape == signal.shape
    assert bool(np.all(envelope >= 0.0))
    assert estimator.nominal_delay_samples(SAMPLE_RATE_HZ) > 0.0


def test_latency_measurement_rejects_a_step_that_goes_the_wrong_way() -> None:
    """Measuring the latency of a step down is refused rather than returned as nonsense."""
    signal = _carrier_step(SAMPLE_RATE_HZ)[::-1].copy()
    step_index = int(STEP_S * SAMPLE_RATE_HZ)
    plateau = slice(step_index + int(0.5 * SAMPLE_RATE_HZ), signal.size)
    with pytest.raises(ValueError):
        measure_latency(MovingAverageEnvelope(0.1), signal, SAMPLE_RATE_HZ, step_index, plateau)
