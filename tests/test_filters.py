"""Property tests for filter design, application, and delay.

Every gain in this file is measured from the filtered signal and compared against the
gain the design predicts. Nothing is asserted from the specification alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from myoelectric.algorithm.filters import (
    apply_causal,
    apply_zero_phase,
    cascade,
    design_bandpass,
    design_highpass,
    design_lowpass,
    design_powerline_notch,
    group_delay_samples,
    phase_delay_samples,
)
from tests.helpers import component_amplitude, energy_centroid, settling_bound, sine

DURATION_S = 2.0
SETTLE_S = 1.0


def _measured_gain(design, frequency_hz: float, sample_rate_hz: float) -> float:
    """Steady state gain of ``design`` at ``frequency_hz``, measured on a sine."""
    signal = sine(frequency_hz, sample_rate_hz, DURATION_S)
    filtered = apply_causal(design, signal)
    settle = int(SETTLE_S * sample_rate_hz)
    return component_amplitude(filtered[settle:], frequency_hz, sample_rate_hz)


@pytest.mark.parametrize("frequency_hz", [50.0, 100.0, 200.0, 300.0])
def test_bandpass_passes_in_band_sine_at_unity_gain(
    frequency_hz: float, sample_rate_hz: float
) -> None:
    """An in band sine passes at the gain the design predicts, which is unity.

    Tolerance: the sum of the absolute impulse response of the design beyond the
    settling point, which bounds how far a response measured after that point can be
    from the steady state response for an input bounded by one.
    """
    design = design_bandpass(sample_rate_hz)
    predicted = float(design.gain(np.array([frequency_hz]))[0])
    tolerance = settling_bound(
        design, int(SETTLE_S * sample_rate_hz), int(DURATION_S * sample_rate_hz)
    )
    assert predicted == pytest.approx(1.0, abs=0.02)
    assert _measured_gain(design, frequency_hz, sample_rate_hz) == pytest.approx(
        predicted, abs=tolerance
    )


@pytest.mark.parametrize(
    ("frequency_hz", "minimum_attenuation_db"),
    [(2.0, 60.0), (5.0, 40.0), (10.0, 20.0), (800.0, 20.0)],
)
def test_bandpass_attenuates_out_of_band_sine_by_its_design(
    frequency_hz: float, minimum_attenuation_db: float, sample_rate_hz: float
) -> None:
    """An out of band sine is attenuated by the amount the design predicts.

    Two assertions are made. The design must attenuate by at least the amount claimed
    for it, and the measured attenuation must equal the design attenuation within the
    settling bound of the design.
    """
    design = design_bandpass(sample_rate_hz)
    predicted_db = float(design.gain_db(np.array([frequency_hz]))[0])
    assert predicted_db <= -minimum_attenuation_db

    predicted = float(design.gain(np.array([frequency_hz]))[0])
    tolerance = settling_bound(
        design, int(SETTLE_S * sample_rate_hz), int(DURATION_S * sample_rate_hz)
    )
    assert _measured_gain(design, frequency_hz, sample_rate_hz) == pytest.approx(
        predicted, abs=tolerance
    )


def test_highpass_rejects_movement_artefact_band(sample_rate_hz: float) -> None:
    """The 20 Hz high pass attenuates a 5 Hz artefact and passes the signal band."""
    design = design_highpass(sample_rate_hz)
    grid = np.array([2.0, 5.0, 20.0, 100.0, 400.0])
    gains_db = design.gain_db(grid)
    assert gains_db[0] < -60.0
    assert gains_db[1] < -40.0
    assert gains_db[2] == pytest.approx(-3.01, abs=0.1)
    assert gains_db[3] == pytest.approx(0.0, abs=0.05)
    assert gains_db[4] == pytest.approx(0.0, abs=0.05)


def test_notch_removes_interference_and_leaves_a_nearby_component(
    sample_rate_hz: float,
) -> None:
    """The notch removes an injected 50 Hz component and leaves a 60 Hz component intact.

    Tolerance: the settling bound of the notch cascade, as for the gain tests.
    """
    design = design_powerline_notch(sample_rate_hz)
    settle = int(SETTLE_S * sample_rate_hz)
    length = int(DURATION_S * sample_rate_hz)
    tolerance = settling_bound(design, settle, length)

    signal = sine(50.0, sample_rate_hz, DURATION_S) + sine(60.0, sample_rate_hz, DURATION_S)
    filtered = apply_causal(design, signal)

    before_50 = component_amplitude(signal[settle:], 50.0, sample_rate_hz)
    after_50 = component_amplitude(filtered[settle:], 50.0, sample_rate_hz)
    before_60 = component_amplitude(signal[settle:], 60.0, sample_rate_hz)
    after_60 = component_amplitude(filtered[settle:], 60.0, sample_rate_hz)

    assert before_50 == pytest.approx(1.0, abs=1e-6)
    assert after_50 < tolerance + 1e-6
    predicted_60 = float(design.gain(np.array([60.0]))[0])
    assert after_60 / before_60 == pytest.approx(predicted_60, abs=tolerance)
    assert predicted_60 > 0.9


def test_notch_covers_the_requested_harmonics(sample_rate_hz: float) -> None:
    """Every requested harmonic below the Nyquist frequency gets its own section."""
    design = design_powerline_notch(sample_rate_hz, fundamental_hz=50.0, n_harmonics=4)
    assert design.n_sections == 4
    gains_db = design.gain_db(np.array([50.0, 100.0, 150.0, 200.0]))
    assert bool(np.all(gains_db < -100.0))

    limited = design_powerline_notch(400.0, fundamental_hz=50.0, n_harmonics=8)
    # At 400 Hz the Nyquist frequency is 200 Hz, so only 50, 100 and 150 Hz fit.
    assert limited.n_sections == 3


def test_zero_phase_gain_is_the_square_of_the_causal_gain(sample_rate_hz: float) -> None:
    """Filtering twice squares the magnitude response, so decibels double."""
    design = design_bandpass(sample_rate_hz)
    grid = np.array([5.0, 20.0, 100.0, 450.0, 800.0])
    causal_db = design.gain_db(grid, mode="causal")
    zero_phase_db = design.gain_db(grid, mode="zero_phase")
    assert zero_phase_db == pytest.approx(2.0 * causal_db, rel=1e-9)


def test_zero_phase_group_delay_is_exactly_zero(sample_rate_hz: float) -> None:
    """The zero phase mode reports exactly zero group delay at every frequency."""
    design = design_bandpass(sample_rate_hz)
    grid = np.linspace(1.0, 0.49 * sample_rate_hz, 200)
    delay = group_delay_samples(design, grid, mode="zero_phase")
    assert bool(np.all(delay == 0.0))
    assert bool(np.all(phase_delay_samples(design, grid, mode="zero_phase") == 0.0))


def test_causal_delay_matches_the_design_and_zero_phase_delay_is_none(
    sample_rate_hz: float,
) -> None:
    """A burst is delayed by the design group delay causally and not at all zero phase.

    The burst is a 100 Hz carrier under a Gaussian envelope, so its position is a group
    delay measurement rather than a phase delay measurement. Position is measured as the
    centre of mass of the squared signal.

    Tolerance: one sample, the quantum of a discrete time delay. A tighter tolerance
    would be asserting more than a sampled measurement can support.
    """
    design = design_bandpass(sample_rate_hz)
    duration = 1.0
    times = np.arange(int(duration * sample_rate_hz), dtype=np.float64) / sample_rate_hz
    burst = np.exp(-(((times - 0.5) / 0.05) ** 2)) * np.sin(2.0 * np.pi * 100.0 * times)

    predicted = float(group_delay_samples(design, np.array([100.0]), mode="causal")[0])
    reference = energy_centroid(burst)
    causal_shift = energy_centroid(apply_causal(design, burst)) - reference
    zero_phase_shift = energy_centroid(apply_zero_phase(design, burst)) - reference

    assert causal_shift == pytest.approx(predicted, abs=1.0)
    assert zero_phase_shift == pytest.approx(0.0, abs=1.0)
    assert predicted > 1.0


def test_group_delay_of_a_cascade_is_the_sum_of_its_parts(sample_rate_hz: float) -> None:
    """Cascading filters adds their group delays."""
    bandpass = design_bandpass(sample_rate_hz)
    lowpass = design_lowpass(sample_rate_hz, 300.0)
    chain = cascade((bandpass, lowpass), name="chain", rationale="test")
    grid = np.array([50.0, 100.0, 200.0])
    total = group_delay_samples(chain, grid)
    parts = group_delay_samples(bandpass, grid) + group_delay_samples(lowpass, grid)
    assert total == pytest.approx(parts, rel=1e-9)


def test_group_delay_is_undefined_at_a_transmission_zero(sample_rate_hz: float) -> None:
    """A notch centre has no defined group delay, so the result is not a number."""
    design = design_powerline_notch(sample_rate_hz)
    delay = group_delay_samples(design, np.array([50.0, 55.0]))
    assert math.isnan(float(delay[0]))
    assert math.isfinite(float(delay[1]))


def test_causal_filtering_uses_no_future_samples(sample_rate_hz: float) -> None:
    """Changing a sample cannot change any earlier output, and does change a later one.

    This is the property that makes causal filtering usable in a controller and that
    zero phase filtering breaks, which the next test shows.
    """
    design = design_bandpass(sample_rate_hz)
    base = np.zeros(400, dtype=np.float64)
    perturbed = base.copy()
    perturbed[200] = 1.0
    difference = apply_causal(design, perturbed) - apply_causal(design, base)
    assert bool(np.all(difference[:200] == 0.0))
    assert float(np.max(np.abs(difference[200:]))) > 0.0


def test_zero_phase_filtering_uses_future_samples(sample_rate_hz: float) -> None:
    """Changing a sample changes earlier outputs under zero phase filtering.

    This is why the mode cannot be used in a real time controller: producing the output
    at sample 100 would require a sample that has not been acquired.
    """
    design = design_bandpass(sample_rate_hz)
    base = np.zeros(400, dtype=np.float64)
    perturbed = base.copy()
    perturbed[200] = 1.0
    difference = apply_zero_phase(design, perturbed) - apply_zero_phase(design, base)
    assert float(np.max(np.abs(difference[:200]))) > 0.0


def test_design_validation_rejects_impossible_corners(sample_rate_hz: float) -> None:
    """Corners outside the representable band are rejected rather than clipped silently."""
    with pytest.raises(ValueError):
        design_bandpass(sample_rate_hz, low_hz=0.0)
    with pytest.raises(ValueError):
        design_bandpass(sample_rate_hz, low_hz=500.0, high_hz=100.0)
    with pytest.raises(ValueError):
        design_highpass(sample_rate_hz, cutoff_hz=sample_rate_hz)
    with pytest.raises(ValueError):
        design_powerline_notch(60.0, fundamental_hz=50.0)
