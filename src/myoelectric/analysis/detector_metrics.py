"""Metrics that make onset detectors comparable.

Four quantities are reported for every detector at every signal to noise ratio.

Detection rate
    Fraction of active trials in which the detector placed an onset within the match
    tolerance of the ground truth onset. This is a binomial proportion estimated from
    ``n`` independent trials, so its standard error is ``sqrt(p (1 - p) / n)`` and is
    reported alongside it. A detection rate quoted without that standard error cannot
    be compared against another one.

False positive rate
    Fraction of resting trials in which the detector declared at least one onset.
    Resting trials contain no contraction at all, so every declaration is an error.
    This is the number that a detection rate is meaningless without.

False positives per second
    Total number of declarations over all resting trials, divided by the total resting
    time. This is the more useful figure for a controller, because a controller runs
    continuously rather than in trials, and because it distinguishes a detector that
    fires once in a resting trial from one that fires thirty times.

Timing bias and its distribution
    Difference between the detected and the true onset over the matched trials, in
    milliseconds, positive when the detection is late. The mean, the standard
    deviation, and the twenty fifth, fiftieth and seventy fifth percentiles are all
    reported, because the distribution of the bias is skewed at low signal to noise
    ratio and a mean on its own hides that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myoelectric.pipeline.detection_sweep import DetectorSweepTrace

__all__ = ["DetectorMetrics", "format_metrics_table", "summarise_sweep"]


@dataclass(frozen=True, slots=True)
class DetectorMetrics:
    """Aggregate performance of one detector at one signal to noise ratio."""

    detector: str
    snr_db: float
    n_trials: int
    detection_rate: float
    detection_rate_stderr: float
    false_positive_rate: float
    false_positive_rate_stderr: float
    false_positives_per_second: float
    bias_mean_ms: float
    bias_sd_ms: float
    bias_p25_ms: float
    bias_median_ms: float
    bias_p75_ms: float


def _binomial_stderr(rate: float, n: int) -> float:
    """Standard error of a proportion estimated from ``n`` independent trials."""
    if n < 1:
        return float("nan")
    return float(np.sqrt(max(rate * (1.0 - rate), 0.0) / n))


def summarise_sweep(trace: DetectorSweepTrace) -> tuple[DetectorMetrics, ...]:
    """Reduce a sweep to one metrics row per detector and signal to noise ratio."""
    rows: list[DetectorMetrics] = []
    for name in trace.detector_names:
        for snr_db in trace.spec.snr_db:
            outcomes = [
                outcome
                for outcome in trace.outcomes
                if outcome.detector == name and outcome.snr_db == snr_db
            ]
            if not outcomes:
                continue
            n = len(outcomes)
            matched = [outcome for outcome in outcomes if outcome.matched]
            detection_rate = len(matched) / n

            rest_hits = [outcome.rest_detections for outcome in outcomes]
            false_positive_rate = sum(1 for hits in rest_hits if hits > 0) / n
            rest_seconds = n * trace.rest_duration_s
            per_second = sum(rest_hits) / rest_seconds if rest_seconds > 0.0 else float("nan")

            errors = np.asarray([outcome.error_ms for outcome in matched], dtype=np.float64)
            if errors.size:
                percentiles = np.percentile(errors, [25.0, 50.0, 75.0])
                bias_mean = float(np.mean(errors))
                bias_sd = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0
                p25, p50, p75 = (float(value) for value in percentiles)
            else:
                bias_mean = bias_sd = p25 = p50 = p75 = float("nan")

            rows.append(
                DetectorMetrics(
                    detector=name,
                    snr_db=snr_db,
                    n_trials=n,
                    detection_rate=detection_rate,
                    detection_rate_stderr=_binomial_stderr(detection_rate, n),
                    false_positive_rate=false_positive_rate,
                    false_positive_rate_stderr=_binomial_stderr(false_positive_rate, n),
                    false_positives_per_second=per_second,
                    bias_mean_ms=bias_mean,
                    bias_sd_ms=bias_sd,
                    bias_p25_ms=p25,
                    bias_median_ms=p50,
                    bias_p75_ms=p75,
                )
            )
    return tuple(rows)


def format_metrics_table(metrics: tuple[DetectorMetrics, ...]) -> str:
    """Render metrics as a Markdown table, ready to paste into a report."""
    header = (
        "| Detector | SNR (dB) | Detection rate | False positive rate | FP per s | "
        "Bias mean (ms) | Bias SD (ms) | Bias p25 (ms) | Bias median (ms) | Bias p75 (ms) |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, rule]
    for row in metrics:
        lines.append(
            f"| {row.detector} | {row.snr_db:.0f} | "
            f"{row.detection_rate:.3f} +/- {row.detection_rate_stderr:.3f} | "
            f"{row.false_positive_rate:.3f} +/- {row.false_positive_rate_stderr:.3f} | "
            f"{row.false_positives_per_second:.3f} | "
            f"{row.bias_mean_ms:.1f} | {row.bias_sd_ms:.1f} | "
            f"{row.bias_p25_ms:.1f} | {row.bias_median_ms:.1f} | {row.bias_p75_ms:.1f} |"
        )
    return "\n".join(lines)
