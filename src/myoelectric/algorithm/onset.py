"""Onset detection for myoelectric signals, three methods behind one protocol.

All three detectors implement :class:`OnsetDetector`, take the same arguments and
return the same :class:`OnsetResult`, so they can be swapped and compared without
changing the calling code. All three are causal: the decision at sample ``n`` uses only
samples up to ``n``, so every one of them can run inside a prosthesis controller.

Simple threshold on a smoothed envelope
    Rectify, low pass filter, and declare an onset when the envelope stays above
    ``mean + k sd`` of a resting baseline for a minimum duration. This is the method
    that Di Fabio (1987) evaluated for computerised onset determination and it remains
    the baseline that newer methods are measured against. Its weakness is that the low
    pass filter needed to make the envelope stable also delays it.

Hodges and Bui
    Rectify, low pass filter at 50 Hz, then compare the mean of a short sliding window
    against the same resting baseline statistic. Hodges and Bui (1996) compared
    computer based onset methods against visual determination and found that a sliding
    window mean with a threshold of one to three standard deviations and a window of
    around 25 ms tracked the visual determination most closely. The sliding window
    makes the test less sensitive to isolated large samples than a pointwise threshold.

Bonato style statistical detector
    Whiten the signal with an autoregressive model fitted to the resting baseline, then
    test successive pairs of whitened samples against a chi squared distribution with
    two degrees of freedom. Bonato, D'Alessio and Knaflitz (1998) derived this test as
    an approximation of the generalised likelihood ratio for a change in the variance
    of a Gaussian process, and paired the test with a rule requiring several of the
    last few pair statistics to exceed the threshold before an onset is declared. The
    detector works on the signal itself rather than on a smoothed envelope, so it is
    not delayed by envelope smoothing. Micera, Sabatini and Dario (1998) describe the
    equivalent generalised likelihood ratio formulation.

Reported index and decision delay are different quantities. ``OnsetResult.onset_indices``
holds the sample at which the onset is judged to have occurred. ``decision_delay_s``
holds the extra time that must elapse before the detector can issue that judgement,
because every detector needs evidence accumulated after the onset. A controller
experiences the sum of the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from myoelectric.algorithm.autoregressive import whiten, yule_walker
from myoelectric.algorithm.filters import apply_causal, design_lowpass

__all__ = [
    "BonatoDetector",
    "EnvelopeThresholdDetector",
    "HodgesBuiDetector",
    "OnsetDetector",
    "OnsetResult",
]


@dataclass(frozen=True, slots=True, eq=False)
class OnsetResult:
    """Outcome of running one detector over one record."""

    detector: str
    onset_indices: tuple[int, ...]
    statistic: NDArray[np.float64]
    threshold: float
    decision_delay_s: float

    @property
    def first_onset_index(self) -> int | None:
        """Index of the earliest onset, or ``None`` when nothing was detected."""
        return self.onset_indices[0] if self.onset_indices else None

    @property
    def n_detections(self) -> int:
        """Number of onsets declared over the whole record."""
        return len(self.onset_indices)


@runtime_checkable
class OnsetDetector(Protocol):
    """Common interface for onset detectors."""

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        ...

    def decision_delay_s(self, sample_rate_hz: float) -> float:
        """Time between the reported onset and the moment the decision can be issued."""
        ...

    def detect(self, x: NDArray[np.float64], sample_rate_hz: float) -> OnsetResult:
        """Locate every onset in ``x``."""
        ...


def _baseline_samples(n_samples: int, sample_rate_hz: float, baseline_s: float) -> int:
    count = round(float(baseline_s) * float(sample_rate_hz))
    if count < 2:
        raise ValueError("the baseline window must contain at least two samples")
    if count >= n_samples:
        raise ValueError("the baseline window does not fit inside the record")
    return count


def _run_starts(
    mask: NDArray[np.bool_],
    min_run: int,
    refractory: int,
    min_silence: int,
    offset: int = 0,
) -> tuple[int, ...]:
    """Start of every run of ``True`` that qualifies as a new onset.

    A run qualifies when it is at least ``min_run`` long, when it starts at least
    ``refractory`` samples after the previous accepted onset, and when the preceding
    stretch of ``False`` was at least ``min_silence`` long.

    The silence requirement is what stops a detector declaring a new onset every time
    its statistic dips below the threshold in the middle of an ongoing contraction. It
    is the second half of the segmentation rule that Bonato, D'Alessio and Knaflitz
    (1998) use: an activation interval begins when the statistic rises and ends when it
    has stayed low for long enough, and only after it has ended can another begin.

    ``offset`` is added to every returned index, which lets a detector that works on a
    decimated statistic map its answer back to the sample grid of the record.
    """
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    accepted: list[int] = []
    last_onset = -refractory - 1
    previous_end: int | None = None
    for start, end in zip(starts, ends, strict=True):
        index = int(start)
        long_enough = end - start >= min_run
        rested = previous_end is None or index - previous_end >= min_silence
        if long_enough:
            previous_end = int(end)
        if not long_enough or not rested:
            continue
        if index - last_onset < refractory:
            continue
        accepted.append(index + offset)
        last_onset = index
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class EnvelopeThresholdDetector:
    """Threshold on a causally smoothed, rectified envelope.

    Args:
        baseline_s: Length of the leading resting segment used to estimate the mean and
            standard deviation of the envelope at rest.
        cutoff_hz: Corner of the envelope low pass filter.
        order: Order of the envelope low pass filter.
        threshold_sd: Number of resting standard deviations above the resting mean at
            which the threshold is placed.
        min_duration_s: Minimum time the envelope must stay above the threshold before
            an onset is declared. This suppresses isolated noise excursions and is the
            main control on the false positive rate.
        refractory_s: Minimum separation between successive declared onsets.
        min_silence_s: Time the envelope must spend below the threshold before another
            onset can be declared. Without it the detector declares a new onset every
            time its statistic dips during an ongoing contraction.
    """

    baseline_s: float = 0.35
    cutoff_hz: float = 8.0
    order: int = 2
    threshold_sd: float = 3.0
    min_duration_s: float = 0.05
    refractory_s: float = 0.3
    min_silence_s: float = 0.1

    @property
    def name(self) -> str:
        """Short identifier used in reports, including the sensitivity setting."""
        return f"envelope-threshold k={self.threshold_sd:g}"

    def decision_delay_s(self, sample_rate_hz: float) -> float:
        """The envelope must stay above the threshold for ``min_duration_s``."""
        del sample_rate_hz
        return self.min_duration_s

    def detect(self, x: NDArray[np.float64], sample_rate_hz: float) -> OnsetResult:
        """Locate every onset in ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        baseline = _baseline_samples(samples.size, sample_rate_hz, self.baseline_s)
        design = design_lowpass(sample_rate_hz, self.cutoff_hz, order=self.order)
        envelope = apply_causal(design, np.abs(samples))
        rest = envelope[:baseline]
        threshold = float(np.mean(rest) + self.threshold_sd * np.std(rest))
        mask = envelope > threshold
        mask[:baseline] = False
        onsets = _run_starts(
            mask,
            min_run=max(1, round(float(self.min_duration_s) * float(sample_rate_hz))),
            refractory=max(1, round(float(self.refractory_s) * float(sample_rate_hz))),
            min_silence=max(1, round(float(self.min_silence_s) * float(sample_rate_hz))),
        )
        return OnsetResult(
            detector=self.name,
            onset_indices=onsets,
            statistic=envelope,
            threshold=threshold,
            decision_delay_s=self.decision_delay_s(sample_rate_hz),
        )


@dataclass(frozen=True, slots=True)
class HodgesBuiDetector:
    """Sliding window mean of a rectified, low pass filtered signal.

    Args:
        baseline_s: Length of the leading resting segment used for the baseline mean
            and standard deviation.
        cutoff_hz: Corner of the low pass filter applied to the rectified signal.
            Hodges and Bui (1996) used 50 Hz.
        order: Order of that low pass filter.
        window_s: Length of the sliding mean. Hodges and Bui report best agreement with
            visual determination near 25 ms.
        threshold_sd: Number of resting standard deviations above the resting mean.
        refractory_s: Minimum separation between successive declared onsets.
        min_silence_s: Time the statistic must spend below the threshold before another
            onset can be declared.
    """

    baseline_s: float = 0.35
    cutoff_hz: float = 50.0
    order: int = 2
    window_s: float = 0.025
    threshold_sd: float = 3.0
    refractory_s: float = 0.3
    min_silence_s: float = 0.1

    @property
    def name(self) -> str:
        """Short identifier used in reports, including the sensitivity setting."""
        return f"hodges-bui k={self.threshold_sd:g}"

    def decision_delay_s(self, sample_rate_hz: float) -> float:
        """The whole sliding window must be observed before the mean can be formed."""
        del sample_rate_hz
        return self.window_s

    def detect(self, x: NDArray[np.float64], sample_rate_hz: float) -> OnsetResult:
        """Locate every onset in ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        baseline = _baseline_samples(samples.size, sample_rate_hz, self.baseline_s)
        window = max(2, round(float(self.window_s) * float(sample_rate_hz)))
        design = design_lowpass(sample_rate_hz, self.cutoff_hz, order=self.order)
        rectified = apply_causal(design, np.abs(samples))
        kernel = np.full(window, 1.0 / window, dtype=np.float64)
        trailing_mean = np.convolve(rectified, kernel)[: rectified.size]
        rest = rectified[:baseline]
        threshold = float(np.mean(rest) + self.threshold_sd * np.std(rest))
        mask = trailing_mean > threshold
        mask[:baseline] = False
        # The trailing mean at sample n summarises samples n - window + 1 to n, so a
        # crossing at n is attributed to the first sample of that window.
        onsets = _run_starts(
            mask,
            min_run=1,
            refractory=max(1, round(float(self.refractory_s) * float(sample_rate_hz))),
            min_silence=max(1, round(float(self.min_silence_s) * float(sample_rate_hz))),
            offset=-(window - 1),
        )
        onsets = tuple(max(0, index) for index in onsets)
        return OnsetResult(
            detector=self.name,
            onset_indices=onsets,
            statistic=trailing_mean,
            threshold=threshold,
            decision_delay_s=self.decision_delay_s(sample_rate_hz),
        )


@dataclass(frozen=True, slots=True)
class BonatoDetector:
    """Chi squared test on pairs of whitened samples, with an ``m`` of ``n`` rule.

    An autoregressive model of order ``ar_order`` is fitted to the resting baseline and
    used to whiten the whole record. Under the hypothesis that the muscle is at rest,
    the whitened samples are independent, zero mean and Gaussian with the baseline
    variance, so the sum of squares of any two of them, divided by that variance, is
    distributed as chi squared with two degrees of freedom. The detection threshold is
    the upper quantile of that distribution at ``false_alarm_probability``, which fixes
    the per test false alarm probability by construction rather than by tuning.

    Args:
        baseline_s: Length of the leading resting segment used to fit the model and to
            estimate the residual variance.
        ar_order: Order of the whitening model.
        false_alarm_probability: Per test false alarm probability used to set the
            threshold.
        required_exceedances: Number of pair statistics within the decision window that
            must exceed the threshold, the ``m`` of the ``m`` of ``n`` rule.
        decision_window_pairs: Length of the decision window in pairs, the ``n``.
        refractory_s: Minimum separation between successive declared onsets.
        min_silence_s: Time the decision rule must stay untriggered before another onset
            can be declared. This is the activation interval rule of the original
            method, in which an interval that has begun must end before another can
            begin.
    """

    baseline_s: float = 0.35
    ar_order: int = 4
    false_alarm_probability: float = 1e-3
    required_exceedances: int = 3
    decision_window_pairs: int = 5
    refractory_s: float = 0.3
    min_silence_s: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.false_alarm_probability < 1.0:
            raise ValueError("false_alarm_probability must lie in (0, 1)")
        if self.required_exceedances < 1:
            raise ValueError("required_exceedances must be at least 1")
        if self.decision_window_pairs < self.required_exceedances:
            raise ValueError("decision_window_pairs must be at least required_exceedances")

    @property
    def name(self) -> str:
        """Short identifier used in reports, including the sensitivity setting."""
        return f"bonato-glr p={self.false_alarm_probability:g}"

    def decision_delay_s(self, sample_rate_hz: float) -> float:
        """Worst case time to accumulate the required exceedances, at 2 samples per pair."""
        return 2.0 * self.decision_window_pairs / sample_rate_hz

    def detect(self, x: NDArray[np.float64], sample_rate_hz: float) -> OnsetResult:
        """Locate every onset in ``x``."""
        samples = np.asarray(x, dtype=np.float64).ravel()
        baseline = _baseline_samples(samples.size, sample_rate_hz, self.baseline_s)
        if baseline <= 2 * self.ar_order:
            raise ValueError("the baseline window is too short for the requested model order")

        model = yule_walker(samples[:baseline], self.ar_order)
        residual = whiten(samples, model)
        rest = residual[self.ar_order : baseline]
        variance = float(np.mean(rest**2))
        if variance <= 0.0:
            raise ValueError("the baseline residual has zero variance")

        n_pairs = residual.size // 2
        pairs = residual[: 2 * n_pairs].reshape(n_pairs, 2)
        statistic = np.asarray(np.sum(pairs**2, axis=1) / variance, dtype=np.float64)
        threshold = float(sp_stats.chi2.ppf(1.0 - self.false_alarm_probability, df=2))

        above = statistic > threshold
        above[: baseline // 2] = False
        kernel = np.ones(self.decision_window_pairs, dtype=np.float64)
        counts = np.convolve(above.astype(np.float64), kernel)[: above.size]
        triggered = counts >= float(self.required_exceedances)

        refractory_pairs = max(1, round(0.5 * float(self.refractory_s) * float(sample_rate_hz)))
        silence_pairs = max(1, round(0.5 * float(self.min_silence_s) * float(sample_rate_hz)))
        window_starts = _run_starts(
            triggered, min_run=1, refractory=refractory_pairs, min_silence=silence_pairs
        )

        onsets: list[int] = []
        for trigger in window_starts:
            first = max(0, trigger - self.decision_window_pairs + 1)
            candidates = np.flatnonzero(above[first : trigger + 1])
            pair_index = first + int(candidates[0]) if candidates.size else trigger
            onsets.append(2 * pair_index)

        # The statistic is defined per pair; repeat it so that the returned array is on
        # the sample grid of the record and can be plotted against it directly.
        per_sample = np.repeat(statistic, 2)
        padded = np.zeros(samples.size, dtype=np.float64)
        padded[: per_sample.size] = per_sample
        return OnsetResult(
            detector=self.name,
            onset_indices=tuple(onsets),
            statistic=padded,
            threshold=threshold,
            decision_delay_s=self.decision_delay_s(sample_rate_hz),
        )
