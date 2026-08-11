"""Markdown tables, checked against the values they claim to be rendering.

Every number in the documentation is copied from one of these tables, so a formatter
that rounded the wrong field or transposed two columns would put a wrong number in the
README without any other test noticing. Each test therefore compares the rendered cell
against the value read straight off the dataclass.
"""

from __future__ import annotations

import numpy as np
import pytest

from myoelectric.algorithm.envelope import LatencyMeasurement
from myoelectric.algorithm.features_freq import frequency_domain_features, welch_spectrum
from myoelectric.algorithm.features_time import time_domain_features
from myoelectric.algorithm.filters import design_bandpass, design_powerline_notch
from myoelectric.analysis.reporting import (
    format_filter_response_table,
    format_frequency_feature_table,
    format_latency_table,
    format_time_feature_table,
)
from tests.helpers import SAMPLE_RATE_HZ, sine

FREQUENCIES = (20.0, 50.0, 100.0, 250.0, 450.0)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_the_filter_table_has_one_row_per_frequency_and_mode() -> None:
    """Two modes over five frequencies is ten rows under one header and one rule."""
    table = format_filter_response_table(design_bandpass(SAMPLE_RATE_HZ), FREQUENCIES)
    lines = table.splitlines()
    assert len(lines) == 2 + 2 * len(FREQUENCIES)
    assert _cells(lines[0]) == ["Frequency (Hz)", "Mode", "Gain (dB)", "Group delay (ms)"]


def test_the_filter_table_reports_the_gain_the_design_has() -> None:
    """The rendered cell has to be the design's own value, not a nearby one."""
    design = design_bandpass(SAMPLE_RATE_HZ)
    table = format_filter_response_table(design, FREQUENCIES, modes=("causal",))
    grid = np.asarray(FREQUENCIES, dtype=np.float64)
    expected = design.gain_db(grid, mode="causal")
    for line, frequency, gain in zip(table.splitlines()[2:], FREQUENCIES, expected, strict=True):
        cells = _cells(line)
        assert float(cells[0]) == pytest.approx(frequency)
        assert cells[1] == "causal"
        assert float(cells[2]) == pytest.approx(float(gain), abs=5e-3)


def test_the_filter_table_reports_zero_delay_for_zero_phase() -> None:
    """A zero phase row that showed a delay would contradict the library's own claim."""
    table = format_filter_response_table(design_bandpass(SAMPLE_RATE_HZ), FREQUENCIES)
    zero_phase = [line for line in table.splitlines()[2:] if _cells(line)[1] == "zero_phase"]
    assert len(zero_phase) == len(FREQUENCIES)
    assert all(float(_cells(line)[3]) == 0.0 for line in zero_phase)


def test_the_filter_table_prints_a_transmission_zero_as_undefined() -> None:
    """Group delay at a notch centre is undefined and must not print as a number."""
    table = format_filter_response_table(
        design_powerline_notch(SAMPLE_RATE_HZ), (50.0,), modes=("causal",)
    )
    assert _cells(table.splitlines()[2])[3] == "nan"


def test_the_time_feature_table_renders_every_feature() -> None:
    """Ten features, each on its own row, under one header and one rule."""
    signal = sine(100.0, SAMPLE_RATE_HZ, 0.25)
    features = time_domain_features(signal, amplitude_threshold=0.1, slope_threshold=1e-6)
    lines = format_time_feature_table(features).splitlines()
    assert len(lines) == 12
    names = [_cells(line)[0] for line in lines[2:]]
    assert names[0] == "Mean absolute value"
    assert names[-1] == "Autoregressive coefficients"


def test_the_time_feature_table_reports_the_values_it_was_given() -> None:
    """Counts are rendered exactly and amplitudes to four decimal places."""
    signal = sine(100.0, SAMPLE_RATE_HZ, 0.25)
    features = time_domain_features(signal, amplitude_threshold=0.1, slope_threshold=1e-6)
    rendered = {
        _cells(line)[0]: _cells(line)[1]
        for line in format_time_feature_table(features).splitlines()[2:]
    }
    assert rendered["Zero crossings"] == str(features.zero_crossings)
    assert rendered["Willison amplitude"] == str(features.willison_amplitude)
    assert float(rendered["Root mean square"]) == pytest.approx(features.root_mean_square, abs=5e-5)
    assert float(rendered["Variance"]) == pytest.approx(features.variance, abs=5e-5)


def test_the_frequency_feature_table_reports_the_values_it_was_given() -> None:
    """Median and mean frequency to two decimals, resolution to three significant figures."""
    signal = sine(100.0, SAMPLE_RATE_HZ, 1.0) + 0.5 * sine(200.0, SAMPLE_RATE_HZ, 1.0)
    spectrum = welch_spectrum(signal, SAMPLE_RATE_HZ)
    features = frequency_domain_features(spectrum)
    rows = {
        _cells(line)[0]: _cells(line)[1]
        for line in format_frequency_feature_table(features).splitlines()[2:]
    }
    assert len(rows) == 7
    assert float(rows["Median frequency (Hz)"]) == pytest.approx(
        features.median_frequency_hz, abs=5e-3
    )
    assert float(rows["Mean frequency (Hz)"]) == pytest.approx(features.mean_frequency_hz, abs=5e-3)
    assert float(rows["Resolution (Hz)"]) == pytest.approx(features.resolution_hz, rel=1e-2)


def test_the_latency_table_has_one_row_per_measurement() -> None:
    """Five columns, and one row for each estimator that was measured."""
    measurements = tuple(
        LatencyMeasurement(
            estimator=name,
            nominal_delay_ms=nominal,
            latency_ms=latency,
            rise_time_ms=rise,
            plateau_ripple_percent=ripple,
        )
        for name, nominal, latency, rise, ripple in (
            ("moving-average-100ms", 49.8, 47.8, 84.6, 15.1),
            ("exponential-50ms", 49.8, 28.1, 96.2, 14.6),
        )
    )
    lines = format_latency_table(measurements).splitlines()
    assert len(lines) == 2 + len(measurements)
    first = _cells(lines[2])
    assert first[0] == "moving-average-100ms"
    assert float(first[1]) == pytest.approx(49.8)
    assert float(first[2]) == pytest.approx(47.8)
    assert float(first[3]) == pytest.approx(84.6)
    assert float(first[4]) == pytest.approx(15.1)
