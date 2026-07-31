"""Property tests for the muscle fatigue demonstration.

The physiological prediction is directional: a sustained contraction lowers the median
frequency of the surface signal. The test therefore asserts a statistically significant
downward trend, using a named statistic on a stated number of degrees of freedom, and
checks that the size of the fall matches the spectral compression that the simulated
slowing of the action potentials predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoelectric.algorithm.filters import cascade, design_bandpass, design_powerline_notch
from myoelectric.analysis.fatigue_stats import analyse_fatigue, format_fatigue_summary
from myoelectric.pipeline.fatigue import FatigueSpec, run_fatigue_protocol

SAMPLE_RATE_HZ = 2000.0
SIGNIFICANCE = 0.01


def _spec(**overrides: object) -> FatigueSpec:
    defaults: dict[str, object] = {
        "duration_s": 40.0,
        "epoch_s": 2.0,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "preprocess": cascade(
            (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
            name="bandpass then notch",
            rationale="Offline analysis, so zero phase filtering is legitimate.",
        ),
        "preprocess_mode": "zero_phase",
    }
    defaults.update(overrides)
    return FatigueSpec(**defaults)  # type: ignore[arg-type]


def test_median_frequency_declines_significantly() -> None:
    """The median frequency trend is negative at the one per cent level, one sided.

    Statistic: the Student t statistic of the ordinary least squares slope of median
    frequency on epoch start time, on ``n - 2`` degrees of freedom. The test is one
    sided because fatigue is predicted to lower the median frequency and not to raise
    it, so a two sided probability would be answering a question that was not asked.
    """
    trace = run_fatigue_protocol(_spec())
    median_trend, _ = analyse_fatigue(trace)

    assert median_trend.n_epochs == trace.n_epochs
    assert median_trend.degrees_of_freedom == trace.n_epochs - 2
    assert median_trend.slope_hz_per_s < 0.0
    assert median_trend.one_sided_p_value < SIGNIFICANCE
    assert median_trend.t_statistic < 0.0
    assert median_trend.is_significant_decline


def test_mean_frequency_declines_with_the_median() -> None:
    """Both spectral centre estimators move in the same direction."""
    trace = run_fatigue_protocol(_spec())
    median_trend, mean_trend = analyse_fatigue(trace)
    assert mean_trend.slope_hz_per_s < 0.0
    assert mean_trend.one_sided_p_value < SIGNIFICANCE
    assert median_trend.normalised_slope_percent_per_s < 0.0


def test_the_size_of_the_fall_matches_the_simulated_compression() -> None:
    """The fitted fall agrees with the ``1 / s`` compression that the model predicts.

    The action potential time constant is scaled from one to ``duration_scale_end``
    across the contraction, and the Hermite Rodriguez magnitude spectrum peaks at
    ``1 / (pi lambda)``, so the spectrum should compress by the reciprocal of that
    scale.

    Tolerance. The predicted ratio is compared against the ratio of the fitted line at
    the last epoch to its value at the first. The uncertainty of that ratio follows from
    the standard error of the slope: propagating it over the fitted span gives
    ``3 * stderr * span / initial``, which is three standard errors of the ratio itself.
    """
    spec = _spec(duration_scale_end=1.35)
    trace = run_fatigue_protocol(spec)
    trend, _ = analyse_fatigue(trace)

    first = float(trace.epoch_start_s[0])
    last = float(trace.epoch_start_s[-1])
    initial = trend.intercept_hz + trend.slope_hz_per_s * first
    final = trend.intercept_hz + trend.slope_hz_per_s * last
    ratio = final / initial
    tolerance = 3.0 * trend.slope_stderr_hz_per_s * (last - first) / initial

    assert ratio == pytest.approx(1.0 / spec.duration_scale_end, abs=tolerance)


def test_no_fatigue_gives_no_trend() -> None:
    """With the potential duration held constant there is no systematic decline.

    This is the control. Without it, a test that only checks for a downward trend would
    also pass on an implementation that always reports one.

    The comparison is made over several seeds and stated as a comparison of magnitudes
    rather than as a single probability, because a single probability compared against a
    fixed level is a knife edge: one control run in a hundred is significant at the one
    per cent level by construction, and which run that is depends on the platform.
    """
    def slope_magnitude(scale: float, seed: int) -> float:
        trace = run_fatigue_protocol(_spec(duration_scale_end=scale, seed=seed))
        return abs(analyse_fatigue(trace)[0].slope_hz_per_s)

    seeds = (11, 22, 33, 44, 55, 66)
    control = [slope_magnitude(1.0, seed) for seed in seeds]
    fatigued = [slope_magnitude(1.35, seed) for seed in seeds]
    assert float(np.median(control)) < 0.2 * float(np.median(fatigued))
    assert sum(1 for value in control if value > min(fatigued)) == 0


def test_amplitude_is_reported_alongside_the_spectrum() -> None:
    """Every epoch carries its root mean square, so amplitude and spectrum can be compared."""
    trace = run_fatigue_protocol(_spec())
    assert trace.root_mean_square.shape == trace.median_frequency_hz.shape
    assert bool(np.all(trace.root_mean_square > 0.0))
    assert bool(np.all(np.diff(trace.epoch_start_s) > 0.0))
    assert "Welch" in trace.spectrum_method


def test_summary_table_reports_the_statistic() -> None:
    """The rendered summary carries the statistic and the degrees of freedom."""
    trace = run_fatigue_protocol(_spec(duration_s=20.0))
    trends = analyse_fatigue(trace)
    table = format_fatigue_summary(trends)
    assert "One sided p" in table
    assert "median frequency" in table
    assert "mean frequency" in table


def test_fatigue_spec_validation() -> None:
    """A protocol too short to fit a trend is rejected."""
    with pytest.raises(ValueError):
        FatigueSpec(duration_s=2.0, epoch_s=1.0)
    with pytest.raises(ValueError):
        FatigueSpec(duration_s=-1.0)
