"""Figures for the traces produced by the pipeline layer.

Every function here builds a :class:`matplotlib.figure.Figure` through the object
oriented interface and attaches an Agg canvas explicitly. The pyplot state machine is
never touched, so these functions work in a headless environment without a backend
having to be selected first, and they hold no global state.

A myoelectric record is a dense trace: two seconds at 2000 Hz is four thousand samples,
and a sustained contraction is sixty times that. Drawing every sample of such a record
costs file size without adding anything a reader can see, because many samples land in
the same column of pixels. :func:`minmax_decimate` reduces a trace to one vertical
extent per column, which is what the eye reads off a dense trace anyway, and is drawn
as a filled band rather than as a line so the cost is one polygon rather than thousands
of segments. Plain subsampling is not used: it would drop the peaks, and the visual
amplitude of an electromyogram is its peaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from myoelectric.algorithm.envelope import LatencyMeasurement
from myoelectric.algorithm.features_freq import PowerSpectrum
from myoelectric.algorithm.filters import FilterDesign
from myoelectric.algorithm.onset import OnsetResult
from myoelectric.analysis.detector_metrics import DetectorMetrics
from myoelectric.analysis.fatigue_stats import FatigueTrend
from myoelectric.pipeline.fatigue import FatigueTrace
from myoelectric.pipeline.generation import SignalTrace

__all__ = [
    "detector_comparison_figure",
    "fatigue_figure",
    "filter_response_figure",
    "latency_figure",
    "minmax_decimate",
    "onset_marks_figure",
    "pareto_frontier",
    "save",
    "signal_overview_figure",
]

# Distinct enough to be told apart in the light and dark backgrounds a README is read
# on, and ordered so that the three onset detectors keep the same colour wherever they
# appear.
_DETECTOR_COLOURS: tuple[str, ...] = ("#c2571a", "#1b7f4b", "#7b3fa0")


def _figure(width: float = 9.0, height: float = 6.0) -> Figure:
    figure = Figure(figsize=(width, height), dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save(figure: Figure, path: Path, dpi: float | None = None) -> Path:
    """Write ``figure`` to ``path``, creating parent directories as needed.

    ``dpi`` overrides the figure's own resolution. The tracked figures under
    ``docs/figures`` are written at a lower resolution than the default, because they
    are displayed at README width and are held to a size budget; a figure written for
    inspection on screen wants the default.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        figure.savefig(path)
    else:
        figure.savefig(path, dpi=dpi)
    return path


def minmax_decimate(
    times_s: NDArray[np.float64], values: NDArray[np.float64], columns: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reduce a dense trace to the minimum and maximum within each of ``columns`` bins.

    Returns the bin centre times, the minimum in each bin, and the maximum in each bin,
    which together describe the vertical extent a reader sees at that horizontal
    position. A trace shorter than ``columns`` is returned unchanged, with the minimum
    and the maximum both equal to the sample, so a caller does not have to special case
    short records.

    The bins are produced by :func:`numpy.array_split`, so they differ in length by at
    most one and every sample belongs to exactly one of them. Splitting into equal bins
    and discarding the remainder would silently truncate the end of the record, which on
    a two second trace at 2000 Hz reduced to 600 columns would drop the last 0.2 s.
    """
    if columns < 1:
        raise ValueError("columns must be at least 1")
    t = np.asarray(times_s, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError("times_s and values must have the same length")
    if y.size == 0:
        raise ValueError("cannot decimate an empty trace")
    if y.size <= columns:
        return t, y, y
    value_bins = np.array_split(y, columns)
    time_bins = np.array_split(t, columns)
    return (
        np.asarray([float(np.mean(block)) for block in time_bins], dtype=np.float64),
        np.asarray([float(np.min(block)) for block in value_bins], dtype=np.float64),
        np.asarray([float(np.max(block)) for block in value_bins], dtype=np.float64),
    )


def signal_overview_figure(trace: SignalTrace, spectrum: PowerSpectrum) -> Figure:
    """Generated record, its components, and its power spectrum."""
    figure = _figure(height=7.5)
    axes = figure.subplots(4, 1)
    times = trace.times_s

    axes[0].plot(times, trace.signal, linewidth=0.5, color="#1f4e79")
    for index in trace.onset_indices:
        axes[0].axvline(times[index], color="#c00000", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("signal")
    axes[0].set_title("Synthetic surface electromyogram, ground truth onset dashed")

    axes[1].plot(times, trace.clean, linewidth=0.5, color="#2e7d32")
    axes[1].set_ylabel("clean")

    axes[2].plot(times, trace.excitation, linewidth=1.0, color="#6a1b9a")
    axes[2].set_ylabel("excitation")
    axes[2].set_xlabel("time (s)")

    axes[3].semilogy(spectrum.frequencies_hz, np.maximum(spectrum.power, 1e-12), color="#1f4e79")
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].set_ylabel("power density")
    axes[3].set_title(spectrum.method, fontsize=8)
    return figure


def filter_response_figure(designs: tuple[FilterDesign, ...], frequencies_hz: np.ndarray) -> Figure:
    """Magnitude and group delay of every design, causal and zero phase."""
    figure = _figure(height=6.5)
    axes = figure.subplots(2, 1, sharex=True)
    grid = np.asarray(frequencies_hz, dtype=np.float64)
    for design in designs:
        axes[0].plot(grid, design.gain_db(grid, mode="causal"), linewidth=1.2, label=design.name)
        axes[1].plot(
            grid, design.group_delay_ms(grid, mode="causal"), linewidth=1.2, label=design.name
        )
    axes[1].axhline(0.0, color="#c00000", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("gain (dB)")
    axes[0].set_ylim(-80.0, 10.0)
    axes[0].legend(fontsize=7, loc="lower center")
    axes[0].set_title(
        "Causal responses. Zero phase filtering gives twice the gain in dB "
        "and exactly zero group delay, marked by the dashed line."
    )
    axes[1].set_ylabel("group delay (ms)")
    axes[1].set_xlabel("frequency (Hz)")
    return figure


def onset_marks_figure(
    times_s: NDArray[np.float64],
    signal: NDArray[np.float64],
    true_index: int,
    results: tuple[OnsetResult, ...],
    zoom_s: tuple[float, float],
    columns: int = 600,
) -> Figure:
    """One conditioned record with the ground truth onset and every detector's answer.

    The upper panel is the whole record, reduced to one vertical extent per column, with
    the region the lower panel expands shaded. The lower panel is that region at full
    sample resolution, with the ground truth first discharge and each detector's first
    declaration drawn as vertical lines.

    What this shows and a column of timing biases does not: at a realistic signal to
    noise ratio the ground truth instant is not visible in the trace. It is the sample
    at which one motor unit discharged, and one discharge does not lift the signal out
    of the noise, so every detector is necessarily late and the disagreement between
    them is smaller than the width of the region in which a human would place the onset
    by eye.
    """
    t = np.asarray(times_s, dtype=np.float64).ravel()
    y = np.asarray(signal, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError("times_s and signal must have the same length")
    low_s, high_s = float(zoom_s[0]), float(zoom_s[1])
    if low_s >= high_s:
        raise ValueError("zoom_s must be an increasing pair")
    inside = (t >= low_s) & (t <= high_s)
    if not bool(np.any(inside)):
        raise ValueError("zoom_s selects no samples")

    figure = _figure(height=5.4)
    axes = figure.subplots(2, 1, height_ratios=(1.0, 1.25))

    centres, floor, ceiling = minmax_decimate(t, y, columns)
    axes[0].fill_between(centres, floor, ceiling, color="#1f4e79", linewidth=0.0)
    axes[0].axvspan(low_s, high_s, color="#f0a202", alpha=0.28, linewidth=0.0, zorder=0)
    axes[0].set_ylabel("conditioned signal")
    axes[0].set_xlim(float(t[0]), float(t[-1]))
    axes[0].set_title(
        "Whole record, shaded where it is expanded below. "
        f"Drawn as {centres.size} columns rather than {t.size} samples.",
        fontsize=10,
    )

    axes[1].plot(t[inside], y[inside], linewidth=0.6, color="#1f4e79")
    axes[1].axvline(
        t[true_index],
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        label="first motor unit discharge (ground truth)",
    )
    # The timing error is formed from the index difference and the sampling interval
    # rather than by subtracting two entries of the time grid. The two agree to within
    # rounding, and the rounding is enough to move a label that lands on a half
    # millisecond into the wrong whole millisecond.
    interval_s = (float(t[-1]) - float(t[0])) / (t.size - 1) if t.size > 1 else 0.0
    for index, result in enumerate(results):
        colour = _DETECTOR_COLOURS[index % len(_DETECTOR_COLOURS)]
        first = result.first_onset_index
        if first is None:
            continue
        error_ms = 1e3 * (first - true_index) * interval_s
        axes[1].axvline(
            t[first],
            color=colour,
            linewidth=1.6,
            label=f"{result.detector}: {error_ms:+.0f} ms",
        )
    axes[1].set_xlim(low_s, high_s)
    axes[1].set_ylabel("conditioned signal")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8, loc="upper left", framealpha=0.92)
    return figure


def detector_comparison_figure(metrics: tuple[DetectorMetrics, ...]) -> Figure:
    """Detection rate, false positive rate, and timing bias against signal to noise ratio."""
    figure = _figure(height=7.0)
    axes = figure.subplots(3, 1, sharex=True)
    names = sorted({row.detector for row in metrics})
    for name in names:
        rows = sorted((row for row in metrics if row.detector == name), key=lambda r: r.snr_db)
        snr = [row.snr_db for row in rows]
        axes[0].errorbar(
            snr,
            [row.detection_rate for row in rows],
            yerr=[row.detection_rate_stderr for row in rows],
            marker="o",
            capsize=3,
            label=name,
        )
        axes[1].errorbar(
            snr,
            [row.false_positive_rate for row in rows],
            yerr=[row.false_positive_rate_stderr for row in rows],
            marker="s",
            capsize=3,
            label=name,
        )
        axes[2].errorbar(
            snr,
            [row.bias_mean_ms for row in rows],
            yerr=[row.bias_sd_ms for row in rows],
            marker="^",
            capsize=3,
            label=name,
        )
    axes[0].set_ylabel("detection rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("false positive rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("timing bias (ms)")
    axes[2].axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("signal to noise ratio (dB)")
    return figure


def fatigue_figure(trace: FatigueTrace, trend: FatigueTrend) -> Figure:
    """Median frequency against time with the fitted trend."""
    figure = _figure(height=5.0)
    axes = figure.subplots(2, 1, sharex=True)
    times = trace.epoch_start_s
    axes[0].plot(times, trace.median_frequency_hz, marker="o", linewidth=1.0, color="#1f4e79")
    axes[0].plot(
        times,
        trend.intercept_hz + trend.slope_hz_per_s * times,
        linewidth=1.5,
        color="#c00000",
        label=(
            f"slope {trend.slope_hz_per_s:.3f} Hz/s, "
            f"t = {trend.t_statistic:.2f}, p = {trend.one_sided_p_value:.1e}"
        ),
    )
    axes[0].set_ylabel("median frequency (Hz)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(True, linewidth=0.4, alpha=0.4)
    axes[0].set_title(
        f"Median frequency falls from {trace.median_frequency_hz[0]:.1f} Hz to "
        f"{trace.median_frequency_hz[-1]:.1f} Hz while amplitude does not fall with it"
    )
    axes[1].plot(times, trace.root_mean_square, marker="s", linewidth=1.0, color="#2e7d32")
    axes[1].set_ylabel("root mean square")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, linewidth=0.4, alpha=0.4)
    return figure


def pareto_frontier(
    measurements: tuple[LatencyMeasurement, ...],
) -> tuple[LatencyMeasurement, ...]:
    """The estimators that no other estimator beats on both latency and ripple.

    An estimator is dominated when another has latency no greater and ripple no
    greater, with at least one strictly smaller. A dominated estimator is one there is
    no reason to choose, and which of them are dominated is the question a table of
    nine rows does not answer by inspection.
    """
    frontier = [
        item
        for item in measurements
        if not any(
            other.latency_ms <= item.latency_ms
            and other.plateau_ripple_percent <= item.plateau_ripple_percent
            and (
                other.latency_ms < item.latency_ms
                or other.plateau_ripple_percent < item.plateau_ripple_percent
            )
            for other in measurements
        )
    ]
    return tuple(sorted(frontier, key=lambda item: item.latency_ms))


def latency_figure(measurements: tuple[LatencyMeasurement, ...]) -> Figure:
    """Plateau ripple against measured latency, with the non dominated frontier drawn.

    The latency axis is logarithmic. The fast estimators are bunched into a few
    milliseconds of each other while the slow ones are a hundred milliseconds away, and
    on a linear axis the interesting half of the family collapses into one corner.
    """
    figure = _figure(height=5.6)
    axes = figure.subplots(1, 1)
    frontier = pareto_frontier(measurements)
    on_frontier = {item.estimator for item in frontier}
    span = max(item.latency_ms for item in measurements)

    if len(frontier) > 1:
        axes.plot(
            [item.latency_ms for item in frontier],
            [item.plateau_ripple_percent for item in frontier],
            linewidth=1.2,
            color="#1f4e79",
            zorder=1,
            label="non dominated frontier",
        )
    for item in measurements:
        best = item.estimator in on_frontier
        axes.scatter(
            item.latency_ms,
            item.plateau_ripple_percent,
            s=52 if best else 40,
            zorder=3,
            color="#1f4e79" if best else "#ffffff",
            edgecolors="#1f4e79" if best else "#999999",
            linewidths=1.2,
        )
        # A dominated point lies above the frontier by construction, so its label goes
        # above it and a frontier label goes below, and the two never meet in the gap
        # between them. The slowest estimators take their labels on the left so that
        # nothing runs off the right hand edge. No marker moves, so the geometry the
        # reader measures off the axes is still exact.
        to_left = item.latency_ms > 0.5 * span
        axes.annotate(
            item.estimator,
            (item.latency_ms, item.plateau_ripple_percent),
            textcoords="offset points",
            xytext=(-8 if to_left else 8, -12 if best else 6),
            horizontalalignment="right" if to_left else "left",
            fontsize=7,
            color="#222222" if best else "#777777",
        )
    axes.set_xscale("log")
    axes.set_xlabel("measured latency to half amplitude (ms), logarithmic")
    axes.set_ylabel("plateau ripple (%)")
    axes.set_title(
        "Smoothing against latency. Hollow markers are dominated: another estimator "
        "is steadier and faster.",
        fontsize=10,
    )
    axes.margins(y=0.14)
    axes.grid(True, which="both", linewidth=0.4, alpha=0.5)
    if len(frontier) > 1:
        axes.legend(fontsize=8, loc="upper right")
    return figure
