"""Text tables that turn traces into the numbers quoted in the documentation."""

from __future__ import annotations

import numpy as np

from myoelectric.algorithm.envelope import LatencyMeasurement
from myoelectric.algorithm.features_freq import FrequencyDomainFeatures
from myoelectric.algorithm.features_time import TimeDomainFeatures
from myoelectric.algorithm.filters import FilterDesign, FilterMode

__all__ = [
    "format_filter_response_table",
    "format_frequency_feature_table",
    "format_latency_table",
    "format_time_feature_table",
]


def format_filter_response_table(
    design: FilterDesign,
    frequencies_hz: tuple[float, ...],
    modes: tuple[FilterMode, ...] = ("causal", "zero_phase"),
) -> str:
    """Gain and group delay of one design at the given frequencies, as Markdown."""
    grid = np.asarray(frequencies_hz, dtype=np.float64)
    header = "| Frequency (Hz) | Mode | Gain (dB) | Group delay (ms) |"
    rule = "| ---: | --- | ---: | ---: |"
    lines = [header, rule]
    for mode in modes:
        gains = design.gain_db(grid, mode=mode)
        delays = design.group_delay_ms(grid, mode=mode)
        for frequency, gain, delay in zip(grid, gains, delays, strict=True):
            lines.append(f"| {frequency:g} | {mode} | {gain:.2f} | {delay:.2f} |")
    return "\n".join(lines)


def format_time_feature_table(features: TimeDomainFeatures) -> str:
    """One time domain feature vector as a Markdown table."""
    slope = ", ".join(f"{value:.4g}" for value in features.mean_absolute_value_slope)
    coefficients = ", ".join(f"{value:.4f}" for value in features.autoregressive_coefficients)
    rows = [
        ("Mean absolute value", f"{features.mean_absolute_value:.4f}"),
        ("Mean absolute value slope", slope),
        ("Zero crossings", str(features.zero_crossings)),
        ("Slope sign changes", str(features.slope_sign_changes)),
        ("Waveform length", f"{features.waveform_length:.2f}"),
        ("Root mean square", f"{features.root_mean_square:.4f}"),
        ("Variance", f"{features.variance:.4f}"),
        ("Integrated electromyogram", f"{features.integrated_emg:.2f}"),
        ("Willison amplitude", str(features.willison_amplitude)),
        ("Autoregressive coefficients", coefficients),
    ]
    lines = ["| Feature | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def format_frequency_feature_table(features: FrequencyDomainFeatures) -> str:
    """One frequency domain feature vector as a Markdown table."""
    rows = [
        ("Median frequency (Hz)", f"{features.median_frequency_hz:.2f}"),
        ("Mean frequency (Hz)", f"{features.mean_frequency_hz:.2f}"),
        ("Spectral moment 0", f"{features.spectral_moment_0:.4g}"),
        ("Spectral moment 1", f"{features.spectral_moment_1:.4g}"),
        ("Spectral moment 2", f"{features.spectral_moment_2:.4g}"),
        ("Root mean square frequency (Hz)", f"{features.root_mean_square_frequency_hz:.2f}"),
        ("Resolution (Hz)", f"{features.resolution_hz:.3g}"),
    ]
    lines = ["| Feature | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def format_latency_table(measurements: tuple[LatencyMeasurement, ...]) -> str:
    """Amplitude estimator latency and ripple as a Markdown table."""
    header = (
        "| Estimator | Nominal delay (ms) | Measured latency (ms) | "
        "Rise time 10-90 (ms) | Plateau ripple (%) |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: |"
    lines = [header, rule]
    for item in measurements:
        lines.append(
            f"| {item.estimator} | {item.nominal_delay_ms:.1f} | {item.latency_ms:.1f} | "
            f"{item.rise_time_ms:.1f} | {item.plateau_ripple_percent:.1f} |"
        )
    return "\n".join(lines)
