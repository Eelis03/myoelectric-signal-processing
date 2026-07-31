"""Regression tier: a recorded reference run compared with a derived numeric tolerance.

Two rules govern what is pinned here and how tightly.

What is pinned. Only quantities that another machine reproduces: feature values computed
from fixed deterministic inputs, filter responses computed in closed form, detector
verdicts on a seeded record, counts, and aggregate rates. Nothing in this project comes
from an iterative solve that might stop at a different point on a different machine, and
nothing that did would be pinned here, because a value from a solve that has not
converged is not a property of the code.

How tightly. Every tolerance below is derived from the measurement rather than from the
difference that happened to be observed when the reference was recorded, and every one
is separated from the arithmetic noise of its own computation by several orders of
magnitude. A tolerance set equal to the observed error passes on the machine that
recorded it and fails on the next one.

Three scales are used.

``_ARITHMETIC``
    Relative tolerance for a deterministic floating point result. The computations here
    involve on the order of a few thousand dependent operations, so their reproduction
    error across platforms is on the order of ``n eps``, that is ``1e-12`` relative.
    ``1e-9`` sits three orders above that and far below any change that would matter.

Timing, in samples
    A threshold crossing is located to the nearest sample, and a difference of one unit
    in the last place in the statistic can move a crossing by one sample. Two samples is
    used, which is twice that bound.

Rates, from the binomial standard error
    A rate estimated from ``n`` trials has standard error ``sqrt(p (1 - p) / n)`` and is
    quantised to ``1 / n``. The tolerance is the larger of three standard errors and one
    quantum, so a rate recorded as zero or one is pinned to within one trial while an
    intermediate rate is allowed its sampling band.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from myoelectric.algorithm.envelope import (
    ExponentialEnvelope,
    LowPassEnvelope,
    MovingAverageEnvelope,
    MovingRmsEnvelope,
)
from myoelectric.algorithm.features_freq import frequency_domain_features, welch_spectrum
from myoelectric.algorithm.features_time import time_domain_features
from myoelectric.algorithm.filters import (
    apply_causal,
    cascade,
    design_bandpass,
    design_highpass,
    design_powerline_notch,
)
from myoelectric.algorithm.onset import (
    BonatoDetector,
    EnvelopeThresholdDetector,
    HodgesBuiDetector,
)
from myoelectric.analysis.detector_metrics import summarise_sweep
from myoelectric.analysis.fatigue_stats import analyse_fatigue
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.detection_sweep import SweepSpec, run_detector_sweep
from myoelectric.pipeline.fatigue import FatigueSpec, run_fatigue_protocol
from myoelectric.pipeline.generation import GenerationSpec, generate
from myoelectric.pipeline.latency import LatencySpec, run_latency_study

SAMPLE_RATE_HZ = 2000.0
SEED = 20260731
_ARITHMETIC = 1e-9
_TIMING_SAMPLES = 2


def _rate_tolerance(recorded: float, n_trials: int) -> float:
    """Tolerance for a recorded rate, from its binomial standard error and quantisation."""
    standard_error = math.sqrt(max(recorded * (1.0 - recorded), 0.0) / n_trials)
    return max(1.0 / n_trials, 3.0 * standard_error)


def _reference_window() -> np.ndarray:
    """A deterministic window: three sinusoids with no random component at all.

    A seeded pseudo random signal would also be reproducible, but a sum of sinusoids has
    the further property that every feature computed from it can be checked against a
    closed form in the property tier, so a disagreement between the two tiers localises
    the fault.
    """
    index = np.arange(2048, dtype=np.float64)
    return np.asarray(
        1.00 * np.sin(2.0 * np.pi * 80.0 * index / SAMPLE_RATE_HZ)
        + 0.50 * np.sin(2.0 * np.pi * 160.0 * index / SAMPLE_RATE_HZ)
        + 0.25 * np.sin(2.0 * np.pi * 240.0 * index / SAMPLE_RATE_HZ),
        dtype=np.float64,
    )


REFERENCE_TIME_DOMAIN = {
    "mean_absolute_value": 0.685460843883706,
    "waveform_length": 450.013731333624,
    "root_mean_square": 0.8099556264443659,
    "variance": 0.6563485995234962,
    "integrated_emg": 1403.8238082738299,
}
REFERENCE_COUNTS = {"zero_crossings": 81, "slope_sign_changes": 163, "willison_amplitude": 981}
REFERENCE_MAV_SLOPE = (
    -0.0009270793835770563,
    0.001725215590338447,
    -0.003485697145264699,
)
REFERENCE_AR = (
    2.4246425652083654,
    -2.142939782217872,
    0.6964378543911343,
    -0.0204296888145567,
)
REFERENCE_FREQUENCY_DOMAIN = {
    "median_frequency_hz": 78.9375,
    "mean_frequency_hz": 102.85714285714288,
    "spectral_moment_0": 0.6562499999999999,
    "spectral_moment_1": 67.5,
    "spectral_moment_2": 8203.5,
}

RESPONSE_GRID_HZ = (5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 450.0, 600.0, 800.0)
REFERENCE_BANDPASS_GAIN_DB = (
    -49.398142,
    -25.086056,
    -3.0103,
    -0.000464,
    -0.0,
    -0.004598,
    -3.0103,
    -17.464519,
    -45.742048,
)
REFERENCE_BANDPASS_GROUP_DELAY_MS = (
    20.680941,
    23.277049,
    31.67567,
    4.463285,
    1.685145,
    1.175881,
    2.013724,
    1.058502,
    0.61856,
)
REFERENCE_HIGHPASS_GAIN_DB = (
    -48.175584,
    -24.107873,
    -3.0103,
    -0.002806,
    -1e-05,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
)

REFERENCE_TRUE_ONSET = 1636
REFERENCE_ONSETS = {
    "envelope-threshold k=3": 1685,
    "hodges-bui k=3": 1629,
    "bonato-glr p=0.001": 1664,
}

SWEEP_TRIALS = 20
REFERENCE_SWEEP = {
    ("envelope-threshold k=3", 0.0): (0.55, 0.0),
    ("envelope-threshold k=3", 10.0): (1.0, 0.0),
    ("hodges-bui k=3", 0.0): (1.0, 0.0),
    ("hodges-bui k=3", 10.0): (1.0, 0.0),
    ("bonato-glr p=0.001", 0.0): (0.3, 0.0),
    ("bonato-glr p=0.001", 10.0): (1.0, 0.0),
}

REFERENCE_FATIGUE_SLOPE_HZ_PER_S = -0.5594800102767835
REFERENCE_FATIGUE_T = -10.150145087829861

REFERENCE_LATENCY_MS = {
    "moving-average-100ms": 47.76428512593896,
    "moving-rms-100ms": 24.883925198200235,
    "lowpass-4Hz-order2": 55.408185974858895,
    "exponential-50ms": 28.063005592676063,
}


def _chain():
    return cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Reference preprocessing chain.",
    )


def test_time_domain_features_match_the_reference_run() -> None:
    """Feature values on a fixed deterministic window are unchanged.

    Tolerance: ``_ARITHMETIC`` relative for the continuous features and exact equality
    for the counts, which are integers and cannot drift by a fraction.
    """
    features = time_domain_features(
        _reference_window(),
        amplitude_threshold=0.2,
        slope_threshold=0.01,
        n_segments=4,
        ar_order=4,
    )
    for name, expected in REFERENCE_TIME_DOMAIN.items():
        assert getattr(features, name) == pytest.approx(expected, rel=_ARITHMETIC)
    for name, expected_count in REFERENCE_COUNTS.items():
        assert getattr(features, name) == expected_count
    assert features.mean_absolute_value_slope == pytest.approx(
        np.array(REFERENCE_MAV_SLOPE), rel=_ARITHMETIC
    )
    assert features.autoregressive_coefficients == pytest.approx(
        np.array(REFERENCE_AR), rel=_ARITHMETIC
    )


def test_frequency_domain_features_match_the_reference_run() -> None:
    """Spectral feature values on a fixed deterministic window are unchanged.

    Tolerance: ``_ARITHMETIC`` relative. The estimate is a fast Fourier transform of a
    fixed input, so its reproduction error is around ``1e-13`` relative, which is four
    orders below this tolerance and four orders below one thousandth of a frequency bin.
    """
    spectrum = welch_spectrum(_reference_window(), SAMPLE_RATE_HZ, segment_s=0.25)
    features = frequency_domain_features(spectrum)
    assert spectrum.resolution_hz == 4.0
    for name, expected in REFERENCE_FREQUENCY_DOMAIN.items():
        assert getattr(features, name) == pytest.approx(expected, rel=_ARITHMETIC)


def test_filter_responses_match_the_reference_run() -> None:
    """Designed gains and group delays are unchanged.

    Tolerance: ``_ARITHMETIC`` relative, with an absolute floor of ``1e-6`` decibels for
    the entries whose recorded value is zero, since a relative tolerance cannot pin a
    zero. Butterworth design is closed form, so nothing here depends on a solver.
    """
    grid = np.asarray(RESPONSE_GRID_HZ, dtype=np.float64)
    bandpass = design_bandpass(SAMPLE_RATE_HZ)
    highpass = design_highpass(SAMPLE_RATE_HZ)

    assert bandpass.gain_db(grid) == pytest.approx(
        np.array(REFERENCE_BANDPASS_GAIN_DB), rel=_ARITHMETIC, abs=1e-6
    )
    assert bandpass.group_delay_ms(grid) == pytest.approx(
        np.array(REFERENCE_BANDPASS_GROUP_DELAY_MS), rel=_ARITHMETIC, abs=1e-6
    )
    assert highpass.gain_db(grid) == pytest.approx(
        np.array(REFERENCE_HIGHPASS_GAIN_DB), rel=_ARITHMETIC, abs=1e-6
    )


def test_detector_verdicts_on_the_reference_record_are_unchanged() -> None:
    """Each detector declares one onset, at the recorded sample.

    Tolerance: ``_TIMING_SAMPLES`` samples on the index, which is twice the amount by
    which a difference of one unit in the last place can move a threshold crossing. The
    number of detections is pinned exactly, because a count is an integer.
    """
    sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=2.0)
    trace = generate(
        GenerationSpec(
            sampling=sampling,
            profile=ContractionProfile.single(0.8, 1.9, 0.6, rise_s=0.02),
            noise=NoiseSpec(snr_db=10.0),
            powerline=PowerlineSpec(),
        ),
        np.random.default_rng(SEED),
    )
    assert trace.onset_indices[0] == pytest.approx(REFERENCE_TRUE_ONSET, abs=_TIMING_SAMPLES)

    filtered = apply_causal(_chain(), trace.signal)
    for detector in (EnvelopeThresholdDetector(), HodgesBuiDetector(), BonatoDetector()):
        result = detector.detect(filtered, SAMPLE_RATE_HZ)
        assert result.n_detections == 1, f"{detector.name} declared {result.onset_indices}"
        assert result.onset_indices[0] == pytest.approx(
            REFERENCE_ONSETS[detector.name], abs=_TIMING_SAMPLES
        )


def test_sweep_aggregate_rates_are_unchanged() -> None:
    """Detection and false positive rates over a seeded sweep are unchanged.

    Tolerance: from the binomial standard error of the recorded rate over the number of
    trials, with a floor of one quantum, as described in the module docstring. A rate
    recorded as zero or one is therefore pinned to within a single trial.
    """
    trace = run_detector_sweep(
        (EnvelopeThresholdDetector(), HodgesBuiDetector(), BonatoDetector()),
        SweepSpec(
            snr_db=(0.0, 10.0),
            n_trials=SWEEP_TRIALS,
            seed=SEED,
            powerline=PowerlineSpec(),
            preprocess=design_bandpass(SAMPLE_RATE_HZ),
        ),
    )
    metrics = {(row.detector, row.snr_db): row for row in summarise_sweep(trace)}
    assert set(metrics) == set(REFERENCE_SWEEP)

    for key, (detection_rate, false_positive_rate) in REFERENCE_SWEEP.items():
        row = metrics[key]
        assert row.n_trials == SWEEP_TRIALS
        assert row.detection_rate == pytest.approx(
            detection_rate, abs=_rate_tolerance(detection_rate, SWEEP_TRIALS)
        )
        assert row.false_positive_rate == pytest.approx(
            false_positive_rate, abs=_rate_tolerance(false_positive_rate, SWEEP_TRIALS)
        )
        assert row.detection_rate_stderr == pytest.approx(
            math.sqrt(max(row.detection_rate * (1.0 - row.detection_rate), 0.0) / SWEEP_TRIALS),
            rel=_ARITHMETIC,
        )


def test_fatigue_trend_is_unchanged() -> None:
    """The recorded median frequency slope and its statistic are unchanged.

    Tolerance: three standard errors of the slope, taken from the regression itself. A
    slope pinned more tightly than its own standard error would be pinning the noise of
    the simulation rather than its trend.
    """
    trace = run_fatigue_protocol(
        FatigueSpec(
            duration_s=40.0,
            epoch_s=2.0,
            sample_rate_hz=SAMPLE_RATE_HZ,
            preprocess=_chain(),
            seed=SEED,
        )
    )
    trend, _ = analyse_fatigue(trace)
    assert trend.slope_hz_per_s == pytest.approx(
        REFERENCE_FATIGUE_SLOPE_HZ_PER_S, abs=3.0 * trend.slope_stderr_hz_per_s
    )
    assert trend.t_statistic == pytest.approx(REFERENCE_FATIGUE_T, rel=0.2)
    assert trend.is_significant_decline


def test_amplitude_estimator_latency_is_unchanged() -> None:
    """The measured latency of each amplitude estimator is unchanged.

    Tolerance: one sample, which is 0.5 ms at this sample rate. The latency is found by
    interpolating a threshold crossing, so a difference of one unit in the last place can
    move it by at most a fraction of a sample; one sample is the next larger quantum.
    """
    trace = run_latency_study(
        (
            MovingAverageEnvelope(0.100),
            MovingRmsEnvelope(0.100),
            LowPassEnvelope(4.0),
            ExponentialEnvelope(0.050),
        ),
        LatencySpec(
            sample_rate_hz=SAMPLE_RATE_HZ,
            duration_s=3.0,
            step_s=1.0,
            preprocess=_chain(),
            seed=SEED,
        ),
    )
    one_sample_ms = 1e3 / SAMPLE_RATE_HZ
    measured = {item.estimator: item.latency_ms for item in trace.measurements}
    assert set(measured) == set(REFERENCE_LATENCY_MS)
    for name, expected in REFERENCE_LATENCY_MS.items():
        assert measured[name] == pytest.approx(expected, abs=one_sample_ms)
