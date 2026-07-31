"""Statistical analysis of the median frequency trend over a sustained contraction.

The trend is quantified by ordinary least squares regression of the per epoch median
frequency on epoch start time. Three numbers are reported.

Slope
    The regression slope in hertz per second, and the same slope divided by the fitted
    value at time zero and expressed in per cent per second. The normalised form is the
    one that is comparable between subjects and between muscles, because absolute
    median frequency depends on electrode placement and on fibre composition.

Test statistic
    The Student t statistic of the slope, ``slope / stderr(slope)``, on ``n - 2``
    degrees of freedom, with the one sided probability of observing a slope this
    negative under the null hypothesis of no trend. The test is one sided because the
    physiological prediction is directional: fatigue lowers median frequency, it does
    not raise it. Reporting a two sided probability for a directional prediction would
    be the wrong test.

Fit quality
    The coefficient of determination, which says how much of the epoch to epoch
    variation the linear trend accounts for. A significant slope with a low coefficient
    of determination is still a real trend, only a noisy one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from myoelectric.pipeline.fatigue import FatigueTrace

__all__ = ["FatigueTrend", "analyse_fatigue", "format_fatigue_summary"]


@dataclass(frozen=True, slots=True)
class FatigueTrend:
    """Linear trend of a per epoch spectral feature against time."""

    feature: str
    n_epochs: int
    initial_hz: float
    final_hz: float
    slope_hz_per_s: float
    slope_stderr_hz_per_s: float
    normalised_slope_percent_per_s: float
    intercept_hz: float
    t_statistic: float
    degrees_of_freedom: int
    one_sided_p_value: float
    r_squared: float

    @property
    def is_significant_decline(self) -> bool:
        """True when the slope is negative at the conventional one per cent level."""
        return self.slope_hz_per_s < 0.0 and self.one_sided_p_value < 0.01


def _trend(
    times_s: NDArray[np.float64], values: NDArray[np.float64], feature: str
) -> FatigueTrend:
    if times_s.size != values.size:
        raise ValueError("times and values must have the same length")
    if times_s.size < 3:
        raise ValueError("at least three epochs are needed to fit a trend")
    result = sp_stats.linregress(times_s, values)
    slope = float(result.slope)
    stderr = float(result.stderr)
    intercept = float(result.intercept)
    t_statistic = slope / stderr if stderr > 0.0 else float("nan")
    degrees_of_freedom = int(times_s.size - 2)
    one_sided = float(sp_stats.t.cdf(t_statistic, df=degrees_of_freedom))
    normalised = 100.0 * slope / intercept if intercept != 0.0 else float("nan")
    return FatigueTrend(
        feature=feature,
        n_epochs=int(times_s.size),
        initial_hz=float(values[0]),
        final_hz=float(values[-1]),
        slope_hz_per_s=slope,
        slope_stderr_hz_per_s=stderr,
        normalised_slope_percent_per_s=normalised,
        intercept_hz=intercept,
        t_statistic=float(t_statistic),
        degrees_of_freedom=degrees_of_freedom,
        one_sided_p_value=one_sided,
        r_squared=float(result.rvalue) ** 2,
    )


def analyse_fatigue(trace: FatigueTrace) -> tuple[FatigueTrend, FatigueTrend]:
    """Fit the median frequency and the mean frequency trends of a fatigue trace."""
    return (
        _trend(trace.epoch_start_s, trace.median_frequency_hz, "median frequency"),
        _trend(trace.epoch_start_s, trace.mean_frequency_hz, "mean frequency"),
    )


def format_fatigue_summary(trends: tuple[FatigueTrend, ...]) -> str:
    """Render fatigue trends as a Markdown table."""
    header = (
        "| Feature | Epochs | Start (Hz) | End (Hz) | Slope (Hz/s) | "
        "Normalised slope (%/s) | t | df | One sided p | R squared |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, rule]
    for trend in trends:
        p_text = f"{trend.one_sided_p_value:.2e}"
        lines.append(
            f"| {trend.feature} | {trend.n_epochs} | {trend.initial_hz:.1f} | "
            f"{trend.final_hz:.1f} | {trend.slope_hz_per_s:.3f} | "
            f"{trend.normalised_slope_percent_per_s:.3f} | {trend.t_statistic:.2f} | "
            f"{trend.degrees_of_freedom} | {p_text} | {trend.r_squared:.3f} |"
        )
    return "\n".join(lines)
