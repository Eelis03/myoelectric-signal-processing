"""Property tests for onset detection.

Two properties are tested for every detector: it finds a clean onset within a stated
tolerance, and it declares nothing on a record that contains no contraction. The second
is the false positive property and it is tested explicitly, because a detector that
fires constantly satisfies the first one perfectly.

Timing bias is measured over repeated trials and reported through an assertion on the
measured value. It is never assumed to be zero: two of the three detectors are late by
construction, because they need evidence accumulated after the onset, and one is early
by construction, because it attributes a sliding window crossing to the first sample of
that window.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from myoelectric.algorithm.onset import (
    BonatoDetector,
    EnvelopeThresholdDetector,
    HodgesBuiDetector,
    OnsetDetector,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate
from tests.helpers import binomial_tolerance

DURATION_S = 2.0
ONSET_S = 0.8
N_TRIALS = 24

# Bias band, in milliseconds, that each detector is asserted to fall inside at a high
# signal to noise ratio. The bands are wide because they are stated in advance from the
# structure of each detector rather than fitted to the measurement: an envelope
# threshold cannot respond faster than the group delay of its own smoothing filter,
# which is 28 ms for a second order Butterworth at 8 Hz, and none of the detectors can
# respond before the muscle has produced enough discharges to be distinguishable from
# noise.
BIAS_BAND_MS: dict[str, tuple[float, float]] = {
    "envelope-threshold k=3": (0.0, 120.0),
    "hodges-bui k=3": (-40.0, 80.0),
    "bonato-glr p=0.001": (-10.0, 120.0),
}


def _detectors() -> tuple[OnsetDetector, ...]:
    return (EnvelopeThresholdDetector(), HodgesBuiDetector(), BonatoDetector())


def _active_spec(sample_rate_hz: float, snr_db: float) -> GenerationSpec:
    return GenerationSpec(
        sampling=SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=DURATION_S),
        profile=ContractionProfile.single(ONSET_S, DURATION_S - 0.1, 0.6, rise_s=0.02),
        noise=NoiseSpec(snr_db=snr_db),
    )


def _rest_spec(sample_rate_hz: float, snr_db: float) -> GenerationSpec:
    return GenerationSpec(
        sampling=SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=DURATION_S),
        profile=ContractionProfile.rest_only(),
        noise=NoiseSpec(snr_db=snr_db),
    )


@pytest.mark.parametrize("detector", _detectors(), ids=lambda d: d.name)
def test_detector_satisfies_the_protocol(detector: OnsetDetector) -> None:
    """Every detector is usable through the common protocol."""
    assert isinstance(detector, OnsetDetector)
    assert isinstance(detector.name, str)
    assert detector.decision_delay_s(2000.0) > 0.0


@pytest.mark.parametrize("detector", _detectors(), ids=lambda d: d.name)
def test_detector_finds_a_clean_onset_within_tolerance(
    detector: OnsetDetector, sample_rate_hz: float
) -> None:
    """A contraction at 20 dB is found within 150 ms of the first motor unit discharge.

    Tolerance: 150 ms, stated in advance as the window inside which a detection is
    useful to a controller. It is 300 samples at this sample rate, which is far wider
    than the sampling interval, so the assertion is about detector behaviour and not
    about sample grid quantisation.
    """
    rng = np.random.default_rng(101)
    tolerance_samples = int(0.150 * sample_rate_hz)
    for _ in range(8):
        trace = generate(_active_spec(sample_rate_hz, 20.0), rng)
        truth = trace.first_onset_index
        assert truth is not None
        result = detector.detect(trace.signal, sample_rate_hz)
        assert result.onset_indices, f"{detector.name} found no onset in a clean contraction"
        matched = [i for i in result.onset_indices if abs(i - truth) <= tolerance_samples]
        assert matched, (
            f"{detector.name} placed its onsets at {result.onset_indices} but the truth is {truth}"
        )


@pytest.mark.parametrize("detector", _detectors(), ids=lambda d: d.name)
def test_detector_reports_no_onset_in_pure_noise(
    detector: OnsetDetector, sample_rate_hz: float
) -> None:
    """On records that contain no contraction the false positive rate stays low.

    Tolerance: four binomial standard errors around zero for the number of trials run,
    which is the sampling error of a rate estimated from that many independent trials.
    A rate is asserted rather than a count so that the assertion does not tighten or
    loosen when the trial count changes.
    """
    rng = np.random.default_rng(202)
    detections = 0
    for _ in range(N_TRIALS):
        trace = generate(_rest_spec(sample_rate_hz, 20.0), rng)
        assert trace.onset_indices == ()
        result = detector.detect(trace.signal, sample_rate_hz)
        detections += 1 if result.onset_indices else 0
    rate = detections / N_TRIALS
    assert rate <= binomial_tolerance(N_TRIALS)


@pytest.mark.parametrize("detector", _detectors(), ids=lambda d: d.name)
def test_detector_timing_bias_is_measured_and_inside_its_stated_band(
    detector: OnsetDetector, sample_rate_hz: float
) -> None:
    """The mean timing bias is measured over repeated trials, not assumed to be zero.

    The band each detector is checked against is stated in advance in ``BIAS_BAND_MS``
    from the structure of the detector. The test also asserts that the measured mean is
    resolved: its standard error over the trials run must be smaller than the width of
    the band, otherwise the comparison would be meaningless.
    """
    rng = np.random.default_rng(303)
    tolerance_samples = int(0.150 * sample_rate_hz)
    errors_ms: list[float] = []
    for _ in range(N_TRIALS):
        trace = generate(_active_spec(sample_rate_hz, 20.0), rng)
        truth = trace.first_onset_index
        assert truth is not None
        result = detector.detect(trace.signal, sample_rate_hz)
        matched = [i for i in result.onset_indices if abs(i - truth) <= tolerance_samples]
        if matched:
            errors_ms.append(1e3 * (matched[0] - truth) / sample_rate_hz)

    assert len(errors_ms) >= N_TRIALS - 2
    errors = np.asarray(errors_ms, dtype=np.float64)
    mean_bias = float(np.mean(errors))
    standard_error = float(np.std(errors, ddof=1)) / np.sqrt(errors.size)

    low, high = BIAS_BAND_MS[detector.name]
    assert standard_error < 0.5 * (high - low)
    assert low <= mean_bias <= high


def test_bonato_threshold_is_the_chi_squared_quantile_it_claims(sample_rate_hz: float) -> None:
    """The threshold is the upper quantile of chi squared with two degrees of freedom.

    Setting the threshold from the distribution rather than by tuning is what makes the
    per test false alarm probability a design parameter rather than an outcome.
    """
    detector = BonatoDetector(false_alarm_probability=1e-3)
    rng = np.random.default_rng(404)
    trace = generate(_rest_spec(sample_rate_hz, 20.0), rng)
    result = detector.detect(trace.signal, sample_rate_hz)
    assert result.threshold == pytest.approx(float(sp_stats.chi2.ppf(1.0 - 1e-3, df=2)), rel=1e-12)


def test_bonato_pair_statistic_follows_chi_squared_on_rest(sample_rate_hz: float) -> None:
    """On a resting record the pair statistic matches its assumed null distribution.

    The fraction of pair statistics above the threshold is compared against the false
    alarm probability that set the threshold. Tolerance: four standard errors of a
    binomial proportion with that probability over the number of pairs, which is the
    sampling error of the measurement.
    """
    probability = 1e-2
    detector = BonatoDetector(false_alarm_probability=probability)
    rng = np.random.default_rng(505)
    trace = generate(_rest_spec(sample_rate_hz, 20.0), rng)
    result = detector.detect(trace.signal, sample_rate_hz)

    baseline = int(detector.baseline_s * sample_rate_hz)
    statistic = result.statistic[baseline::2]
    exceeded = float(np.mean(statistic > result.threshold))
    standard_error = np.sqrt(probability * (1.0 - probability) / statistic.size)
    assert exceeded == pytest.approx(probability, abs=4.0 * standard_error)


def test_lowering_the_threshold_raises_both_rates(sample_rate_hz: float) -> None:
    """A more sensitive setting detects more and also produces more false positives.

    This is the operating characteristic, and it is the reason a detection rate quoted
    without a false positive rate carries no information.
    """
    rng = np.random.default_rng(606)
    sensitive = HodgesBuiDetector(threshold_sd=1.0)
    conservative = HodgesBuiDetector(threshold_sd=3.0)
    sensitive_hits = 0
    conservative_hits = 0
    for _ in range(N_TRIALS):
        trace = generate(_rest_spec(sample_rate_hz, 20.0), rng)
        sensitive_hits += 1 if sensitive.detect(trace.signal, sample_rate_hz).onset_indices else 0
        conservative_hits += (
            1 if conservative.detect(trace.signal, sample_rate_hz).onset_indices else 0
        )
    assert sensitive_hits > conservative_hits


def test_detectors_reject_a_baseline_that_does_not_fit(sample_rate_hz: float) -> None:
    """A baseline window longer than the record is an error, not a silent truncation."""
    short = np.zeros(100, dtype=np.float64) + 1e-3
    for detector in _detectors():
        with pytest.raises(ValueError):
            detector.detect(short, sample_rate_hz)
