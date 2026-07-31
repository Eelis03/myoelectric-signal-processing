"""Figures for the traces produced by the pipeline layer.

Every function here builds a :class:`matplotlib.figure.Figure` through the object
oriented interface and attaches an Agg canvas explicitly. The pyplot state machine is
never touched, so these functions work in a headless environment without a backend
having to be selected first, and they hold no global state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from myoelectric.algorithm.envelope import LatencyMeasurement
from myoelectric.algorithm.features_freq import PowerSpectrum
from myoelectric.algorithm.filters import FilterDesign
from myoelectric.analysis.detector_metrics import DetectorMetrics
from myoelectric.analysis.fatigue_stats import FatigueTrend
from myoelectric.pipeline.fatigue import FatigueTrace
from myoelectric.pipeline.generation import SignalTrace

__all__ = [
    "detector_comparison_figure",
    "fatigue_figure",
    "filter_response_figure",
    "latency_figure",
    "save",
    "signal_overview_figure",
]


def _figure(width: float = 9.0, height: float = 6.0) -> Figure:
    figure = Figure(figsize=(width, height), dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save(figure: Figure, path: Path) -> Path:
    """Write ``figure`` to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


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


def filter_response_figure(
    designs: tuple[FilterDesign, ...], frequencies_hz: np.ndarray
) -> Figure:
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
    axes[0].set_title("Causal responses. Zero phase filtering gives twice the gain in dB "
                      "and exactly zero group delay, marked by the dashed line.")
    axes[1].set_ylabel("group delay (ms)")
    axes[1].set_xlabel("frequency (Hz)")
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
    axes[0].legend(fontsize=8)
    axes[1].plot(times, trace.root_mean_square, marker="s", linewidth=1.0, color="#2e7d32")
    axes[1].set_ylabel("root mean square")
    axes[1].set_xlabel("time (s)")
    return figure


def latency_figure(measurements: tuple[LatencyMeasurement, ...]) -> Figure:
    """Plateau ripple against measured latency for every amplitude estimator."""
    figure = _figure(height=5.0)
    axes = figure.subplots(1, 1)
    for item in measurements:
        axes.scatter(item.latency_ms, item.plateau_ripple_percent, s=45)
        axes.annotate(
            item.estimator,
            (item.latency_ms, item.plateau_ripple_percent),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    axes.set_xlabel("measured latency to half amplitude (ms)")
    axes.set_ylabel("plateau ripple (%)")
    axes.set_title("Smoothing against latency for proportional control")
    axes.grid(True, linewidth=0.4, alpha=0.5)
    return figure
