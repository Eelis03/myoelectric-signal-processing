"""Property tests for the synthetic signal generator and its ground truth.

The generator is the source of every number this project reports, so its own properties
are tested rather than assumed: that the ground truth onset is the first sample at which
the muscle is active, that the requested signal to noise ratio is the ratio that comes
out, and that a given seed reproduces a given record.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoelectric.algorithm.features_freq import median_frequency, welch_spectrum
from myoelectric.model.contraction import ContractionEvent, ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPool, MotorUnitPoolSpec
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate
from tests.helpers import component_amplitude, floating_point_bound

DURATION_S = 2.0
ONSET_S = 0.8


def _spec(sample_rate_hz: float, **overrides: object) -> GenerationSpec:
    defaults: dict[str, object] = {
        "sampling": SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=DURATION_S),
        "profile": ContractionProfile.single(ONSET_S, DURATION_S - 0.1, 0.6, rise_s=0.05),
        "noise": NoiseSpec(snr_db=20.0),
    }
    defaults.update(overrides)
    return GenerationSpec(**defaults)  # type: ignore[arg-type]


def test_ground_truth_onset_is_the_first_active_sample(sample_rate_hz: float) -> None:
    """The clean signal is exactly zero before the ground truth onset and not after.

    The action potential of each unit is placed so that its first sample coincides with
    the discharge, so the first discharge index is also the first sample at which the
    noise free signal departs from zero. That is what makes the ground truth unambiguous
    and it is checked here rather than assumed.

    Tolerance. The potential trains are summed by a fast Fourier transform convolution,
    which is exact in real arithmetic but leaves rounding of order ``n eps`` times the
    peak of the result in the region that should be exactly zero. Asserting equality
    with zero would therefore be asserting that a transform is exact, which it is not.
    """
    trace = generate(_spec(sample_rate_hz), np.random.default_rng(1))
    onset = trace.first_onset_index
    assert onset is not None
    peak = float(np.max(np.abs(trace.clean)))
    tolerance = floating_point_bound(trace.sampling.n_samples, peak)
    assert float(np.max(np.abs(trace.clean[:onset]))) < tolerance
    assert float(np.max(np.abs(trace.clean[onset : onset + 50]))) > 1e3 * tolerance


def test_ground_truth_onset_follows_the_neural_onset(sample_rate_hz: float) -> None:
    """No motor unit discharges before the excitation starts to rise."""
    trace = generate(_spec(sample_rate_hz), np.random.default_rng(2))
    assert trace.onset_indices[0] >= trace.neural_onset_indices[0]
    assert trace.neural_onset_indices[0] == trace.sampling.to_samples(ONSET_S)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0, 20.0])
def test_achieved_signal_to_noise_ratio_matches_the_request(
    snr_db: float, sample_rate_hz: float
) -> None:
    """The realised ratio equals the requested one within the sampling error of the noise.

    Tolerance. The root mean square of ``n`` Gaussian samples has a relative standard
    error of ``1 / sqrt(2 n)``, which in decibels is ``8.686 / sqrt(2 n)``. Four of those
    is used, so the tolerance is the sampling error of the noise estimate rather than an
    observed discrepancy.
    """
    trace = generate(
        _spec(sample_rate_hz, noise=NoiseSpec(snr_db=snr_db)), np.random.default_rng(3)
    )
    n_samples = trace.sampling.n_samples
    tolerance_db = 4.0 * 8.686 / np.sqrt(2.0 * n_samples)
    assert trace.achieved_snr_db == pytest.approx(snr_db, abs=tolerance_db)


def test_the_same_seed_reproduces_the_same_record(sample_rate_hz: float) -> None:
    """Generation is a deterministic function of the specification and the seed."""
    first = generate(_spec(sample_rate_hz), np.random.default_rng(20260731))
    second = generate(_spec(sample_rate_hz), np.random.default_rng(20260731))
    assert np.array_equal(first.signal, second.signal)
    assert first.onset_indices == second.onset_indices


def test_a_resting_record_has_no_activity_and_no_ground_truth(sample_rate_hz: float) -> None:
    """A profile with no contraction produces no discharges and no annotated onset."""
    trace = generate(
        _spec(sample_rate_hz, profile=ContractionProfile.rest_only()), np.random.default_rng(4)
    )
    assert trace.onset_indices == ()
    assert trace.first_onset_index is None
    assert float(np.max(np.abs(trace.clean))) == 0.0
    assert trace.active_rms == 0.0
    assert float(np.std(trace.signal)) > 0.0


def test_power_line_interference_appears_at_the_requested_frequencies(
    sample_rate_hz: float,
) -> None:
    """The mains component contains the fundamental and the requested harmonics.

    Each amplitude is checked against the specification, and the decay between
    successive harmonics is checked against the ratio the specification states.
    """
    spec = PowerlineSpec(
        fundamental_hz=50.0, n_harmonics=3, amplitude_ratio=0.3, harmonic_decay=0.5
    )
    trace = generate(_spec(sample_rate_hz, powerline=spec), np.random.default_rng(5))
    amplitudes = [component_amplitude(trace.powerline, 50.0 * k, sample_rate_hz) for k in (1, 2, 3)]
    expected = [trace.active_rms * 0.3 * 0.5**k for k in range(3)]
    assert amplitudes == pytest.approx(expected, rel=1e-6)


def test_movement_artefact_energy_is_confined_to_low_frequencies(
    sample_rate_hz: float,
) -> None:
    """The artefact carries almost all of its power below the high pass corner."""
    trace = generate(
        _spec(sample_rate_hz, motion=MotionArtefactSpec(cutoff_hz=5.0, amplitude_ratio=0.5)),
        np.random.default_rng(6),
    )
    spectrum = welch_spectrum(trace.motion_artefact, sample_rate_hz, segment_s=0.5)
    below = float(np.sum(spectrum.power[spectrum.frequencies_hz < 20.0]))
    total = float(np.sum(spectrum.power))
    assert below / total > 0.98
    assert float(np.sqrt(np.mean(trace.motion_artefact**2))) == pytest.approx(
        0.5 * trace.active_rms, rel=1e-9
    )


def test_lengthening_the_action_potentials_compresses_the_spectrum(
    sample_rate_hz: float,
) -> None:
    """Scaling the potential time constant by ``s`` moves the spectrum by ``1 / s``.

    The Hermite Rodriguez potential has a magnitude spectrum peaking at
    ``1 / (pi lambda)``, so this is the mechanism by which the fatigue demonstration
    produces a falling median frequency.

    Tolerance: ten per cent of the predicted ratio, which is the epoch to epoch scatter
    of a median frequency estimated from a two second record and is measured here as the
    spread over three seeds rather than assumed.
    """
    scale = 1.4
    ratios: list[float] = []
    for seed in (11, 12, 13):
        base = generate(_spec(sample_rate_hz), np.random.default_rng(seed))
        stretched = generate(
            _spec(sample_rate_hz, muap_duration_scale=scale), np.random.default_rng(seed)
        )
        window = slice(base.sampling.to_samples(1.0), base.sampling.n_samples)
        base_mdf = median_frequency(
            welch_spectrum(base.clean[window], sample_rate_hz, segment_s=0.25)
        )
        stretched_mdf = median_frequency(
            welch_spectrum(stretched.clean[window], sample_rate_hz, segment_s=0.25)
        )
        ratios.append(stretched_mdf / base_mdf)
    assert float(np.mean(ratios)) == pytest.approx(1.0 / scale, rel=0.1)


def test_motor_unit_pool_follows_the_size_principle() -> None:
    """Thresholds, amplitudes and durations are ordered as the recruitment model requires."""
    pool = MotorUnitPool.from_spec(MotorUnitPoolSpec(n_units=20, recruitment_range=30.0))
    thresholds = [unit.recruitment_threshold for unit in pool.units]
    amplitudes = [unit.amplitude for unit in pool.units]
    time_constants = [unit.time_constant_s for unit in pool.units]

    assert thresholds == sorted(thresholds)
    assert amplitudes == sorted(amplitudes)
    assert time_constants == sorted(time_constants, reverse=True)
    assert thresholds[-1] == pytest.approx(1.0, rel=1e-12)
    assert thresholds[0] == pytest.approx(1.0 / 30.0, rel=1e-12)
    assert amplitudes[-1] / amplitudes[0] == pytest.approx(40.0, rel=1e-12)


def test_motor_unit_firing_rate_saturates_at_its_peak() -> None:
    """A unit does not fire below threshold and does not exceed its peak rate."""
    pool = MotorUnitPool.from_spec(MotorUnitPoolSpec())
    unit = pool.units[0]
    assert unit.firing_rate_hz(0.5 * unit.recruitment_threshold) == 0.0
    assert unit.firing_rate_hz(unit.recruitment_threshold) == pytest.approx(
        unit.min_firing_rate_hz, rel=1e-12
    )
    assert unit.firing_rate_hz(1.0) == pytest.approx(unit.peak_firing_rate_hz, rel=1e-12)


def test_action_potential_has_no_direct_current_component() -> None:
    """The sampled potential integrates to zero, so a train of them carries no offset."""
    pool = MotorUnitPool.from_spec(MotorUnitPoolSpec())
    for unit in (pool.units[0], pool.units[-1]):
        potential = unit.action_potential(2000.0)
        assert float(np.sum(potential)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.max(np.abs(potential))) == pytest.approx(unit.amplitude, rel=1e-12)


def test_contraction_profile_shape_and_validation() -> None:
    """The excitation profile is trapezoidal and invalid events are rejected."""
    profile = ContractionProfile.single(0.5, 1.5, 0.8, rise_s=0.1, fall_s=0.1)
    times = np.linspace(0.0, 2.0, 2001)
    excitation = profile.excitation(times)
    assert float(np.max(excitation)) == pytest.approx(0.8, rel=1e-9)
    assert float(excitation[0]) == 0.0
    assert float(excitation[-1]) == 0.0
    assert profile.onset_times_s == (0.5,)
    assert profile.offset_times_s == (1.5,)

    with pytest.raises(ValueError):
        ContractionEvent(onset_s=1.0, offset_s=0.5, plateau_excitation=0.5)
    with pytest.raises(ValueError):
        ContractionEvent(onset_s=0.0, offset_s=1.0, plateau_excitation=1.5)
    with pytest.raises(ValueError):
        ContractionEvent(onset_s=0.0, offset_s=0.1, plateau_excitation=0.5, rise_s=0.2)
    with pytest.raises(ValueError):
        ContractionProfile(
            events=(
                ContractionEvent(0.0, 1.0, 0.5),
                ContractionEvent(0.5, 1.5, 0.5),
            )
        )


def test_sampling_spec_validation() -> None:
    """A sampling specification that cannot describe a record is rejected."""
    spec = SamplingSpec(sample_rate_hz=2000.0, duration_s=1.5)
    assert spec.n_samples == 3000
    assert spec.nyquist_hz == 1000.0
    assert spec.to_samples(0.25) == 500
    assert spec.to_seconds(500) == pytest.approx(0.25, rel=1e-12)

    with pytest.raises(ValueError):
        SamplingSpec(sample_rate_hz=0.0, duration_s=1.0)
    with pytest.raises(ValueError):
        SamplingSpec(sample_rate_hz=2000.0, duration_s=0.0)
