"""Figures, and the display reductions they depend on.

A figure test cannot assert that a picture is good. It can assert the things that make
a picture honest: that a decimation for display keeps the extremes it is meant to keep
and drops no part of the record, that the frontier drawn through a scatter is the
frontier, that a marked onset is drawn where the detector put it, and that the file
that comes out is a PNG of the size the budget assumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from myoelectric.algorithm.envelope import (
    ExponentialEnvelope,
    LatencyMeasurement,
    LowPassEnvelope,
    MovingAverageEnvelope,
)
from myoelectric.algorithm.features_freq import welch_spectrum
from myoelectric.algorithm.filters import (
    apply_causal,
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
from myoelectric.analysis.figures import (
    detector_comparison_figure,
    fatigue_figure,
    filter_response_figure,
    latency_figure,
    minmax_decimate,
    onset_marks_figure,
    pareto_frontier,
    save,
    signal_overview_figure,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.detection_sweep import SweepSpec, run_detector_sweep
from myoelectric.pipeline.fatigue import FatigueSpec, run_fatigue_protocol
from myoelectric.pipeline.generation import GenerationSpec, SignalTrace, generate
from tests.helpers import SAMPLE_RATE_HZ

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def record() -> SignalTrace:
    """One two second record with a contraction at 0.8 s, used by several figures."""
    sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=2.0)
    return generate(
        GenerationSpec(
            sampling=sampling,
            profile=ContractionProfile.single(0.8, 1.8, 0.5, rise_s=0.02, fall_s=0.1),
            noise=NoiseSpec(snr_db=10.0),
            powerline=PowerlineSpec(),
        ),
        np.random.default_rng(20260731),
    )


def _measurement(name: str, latency_ms: float, ripple: float) -> LatencyMeasurement:
    return LatencyMeasurement(
        estimator=name,
        nominal_delay_ms=latency_ms,
        latency_ms=latency_ms,
        rise_time_ms=2.0 * latency_ms,
        plateau_ripple_percent=ripple,
    )


def test_decimation_keeps_the_extremes_of_the_whole_trace() -> None:
    """The point of a minimum and maximum reduction is that no peak is lost."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=4000)
    times = np.arange(values.size, dtype=np.float64) / SAMPLE_RATE_HZ
    _, low, high = minmax_decimate(times, values, 600)
    assert float(np.min(low)) == pytest.approx(float(np.min(values)))
    assert float(np.max(high)) == pytest.approx(float(np.max(values)))


def test_decimation_returns_exactly_the_requested_number_of_columns() -> None:
    """A column count that drifts would make the drawn width unpredictable."""
    values = np.arange(4000, dtype=np.float64)
    times = values / SAMPLE_RATE_HZ
    centres, low, high = minmax_decimate(times, values, 600)
    assert centres.size == low.size == high.size == 600


def test_decimation_covers_the_end_of_a_record_that_does_not_divide_evenly() -> None:
    """Equal bins with the remainder discarded would silently truncate the record.

    Four thousand samples in six hundred columns leaves four hundred over. Splitting
    into equal bins and dropping them would end the drawn trace a fifth of a second
    early, which on a plot with an axis running to the full duration reads as a signal
    that stopped.
    """
    values = np.arange(4000, dtype=np.float64)
    times = values / SAMPLE_RATE_HZ
    centres, _, high = minmax_decimate(times, values, 600)
    assert float(high[-1]) == pytest.approx(float(values[-1]))
    assert float(centres[-1]) < float(times[-1])
    assert float(times[-1]) - float(centres[-1]) < 2.0 * (float(times[1]) - float(times[0])) * 7


def test_decimation_is_ordered_low_below_high() -> None:
    """A band drawn with its edges crossed would be nonsense."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=1000)
    times = np.arange(values.size, dtype=np.float64)
    _, low, high = minmax_decimate(times, values, 100)
    assert bool(np.all(low <= high))


def test_a_trace_shorter_than_the_column_count_is_returned_untouched() -> None:
    """Reducing four samples to six hundred columns would invent detail."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, -2.0, 3.0, 0.5])
    centres, low, high = minmax_decimate(times, values, 600)
    assert np.array_equal(centres, times)
    assert np.array_equal(low, values)
    assert np.array_equal(high, values)


@pytest.mark.parametrize(
    ("times", "values", "columns", "message"),
    [
        (np.zeros(4), np.zeros(4), 0, "columns must be at least 1"),
        (np.zeros(4), np.zeros(5), 10, "same length"),
        (np.zeros(0), np.zeros(0), 10, "empty trace"),
    ],
)
def test_decimation_rejects_inputs_it_cannot_reduce(
    times: np.ndarray, values: np.ndarray, columns: int, message: str
) -> None:
    """Each of these would otherwise produce a plot of something that is not the data."""
    with pytest.raises(ValueError, match=message):
        minmax_decimate(times, values, columns)


def test_the_frontier_excludes_a_point_beaten_on_both_axes() -> None:
    """A dominated estimator is slower and noisier than one already on the list."""
    fast = _measurement("fast", 20.0, 20.0)
    steady = _measurement("steady", 100.0, 5.0)
    beaten = _measurement("beaten", 120.0, 22.0)
    frontier = pareto_frontier((fast, steady, beaten))
    assert [item.estimator for item in frontier] == ["fast", "steady"]


def test_the_frontier_is_returned_in_latency_order() -> None:
    """It is drawn as a line, so an unordered frontier would zigzag."""
    items = (
        _measurement("c", 100.0, 5.0),
        _measurement("a", 20.0, 20.0),
        _measurement("b", 50.0, 10.0),
    )
    assert [item.estimator for item in pareto_frontier(items)] == ["a", "b", "c"]


def test_a_point_tied_on_one_axis_and_worse_on_the_other_is_dominated() -> None:
    """Equal latency with more ripple is never a reason to choose an estimator."""
    frontier = pareto_frontier((_measurement("keep", 30.0, 8.0), _measurement("drop", 30.0, 9.0)))
    assert [item.estimator for item in frontier] == ["keep"]


def test_every_estimator_is_on_the_frontier_when_none_dominates() -> None:
    """A strictly decreasing trade off has nothing to discard."""
    items = (_measurement("a", 10.0, 30.0), _measurement("b", 20.0, 20.0))
    assert len(pareto_frontier(items)) == 2


def test_the_latency_figure_marks_the_frontier_and_the_dominated_points() -> None:
    """Nine annotations, one per estimator, and a line through the frontier only."""
    items = (
        _measurement("fast", 20.0, 20.0),
        _measurement("steady", 100.0, 5.0),
        _measurement("beaten", 120.0, 22.0),
    )
    figure = latency_figure(items)
    axes = figure.axes[0]
    assert len(axes.texts) == len(items)
    assert {text.get_text() for text in axes.texts} == {"fast", "steady", "beaten"}
    assert axes.get_xscale() == "log"
    frontier_lines = [line for line in axes.lines if line.get_label() == "non dominated frontier"]
    assert len(frontier_lines) == 1
    assert list(frontier_lines[0].get_xdata()) == [20.0, 100.0]


def test_the_latency_figure_draws_no_frontier_line_for_a_single_point() -> None:
    """A frontier of one point is a point, and joining it to itself draws nothing."""
    figure = latency_figure((_measurement("only", 30.0, 8.0),))
    assert figure.axes[0].get_legend() is None


def test_the_onset_figure_marks_every_detector_where_it_fired(record: SignalTrace) -> None:
    """A mark drawn anywhere but at the reported index would misreport the detector."""
    conditioned = apply_causal(design_bandpass(SAMPLE_RATE_HZ), record.signal)
    detectors = (EnvelopeThresholdDetector(), HodgesBuiDetector(), BonatoDetector())
    results = tuple(detector.detect(conditioned, SAMPLE_RATE_HZ) for detector in detectors)
    truth = record.onset_indices[0]

    figure = onset_marks_figure(
        record.times_s, conditioned, truth, results, zoom_s=(0.7, 1.0)
    )
    assert len(figure.axes) == 2
    zoom = figure.axes[1]
    assert zoom.get_xlim() == pytest.approx((0.7, 1.0))

    # Every labelled line in the zoom panel is a vertical mark; the trace itself is
    # drawn without a label, so filtering on the label separates the two.
    marks = [line for line in zoom.lines if not str(line.get_label()).startswith("_")]
    drawn = sorted(float(line.get_xdata()[0]) for line in marks)
    expected = sorted(
        [float(record.times_s[truth])]
        + [
            float(record.times_s[result.first_onset_index])
            for result in results
            if result.first_onset_index is not None
        ]
    )
    assert drawn == pytest.approx(expected)


def test_the_onset_figure_labels_each_mark_with_its_signed_error(record: SignalTrace) -> None:
    """The legend has to carry the number, or the reader is measuring pixels."""
    conditioned = apply_causal(design_bandpass(SAMPLE_RATE_HZ), record.signal)
    detector = HodgesBuiDetector()
    result = detector.detect(conditioned, SAMPLE_RATE_HZ)
    truth = record.onset_indices[0]
    assert result.first_onset_index is not None
    expected_ms = 1e3 * (result.first_onset_index - truth) / SAMPLE_RATE_HZ

    figure = onset_marks_figure(
        record.times_s, conditioned, truth, (result,), zoom_s=(0.7, 1.0)
    )
    labels = [text.get_text() for text in figure.axes[1].get_legend().get_texts()]
    assert any("ground truth" in label for label in labels)
    marked = [label for label in labels if label.startswith(detector.name)]
    assert len(marked) == 1
    # Compared as a number rather than as a string, so that a value landing on a half
    # millisecond does not turn the test into a check on rounding direction.
    shown_ms = float(marked[0].rsplit(": ", 1)[1].removesuffix(" ms"))
    assert shown_ms == pytest.approx(expected_ms, abs=0.5)


def test_the_onset_figure_reduces_the_context_panel(record: SignalTrace) -> None:
    """The whole record is drawn as columns, which is the saving the budget relies on."""
    figure = onset_marks_figure(record.times_s, record.signal, 1600, (), zoom_s=(0.7, 1.0))
    title = figure.axes[0].get_title()
    assert "600 columns" in title
    assert f"{record.signal.size} samples" in title


@pytest.mark.parametrize(
    ("zoom", "message"),
    [((1.0, 0.5), "increasing pair"), ((5.0, 6.0), "selects no samples")],
)
def test_the_onset_figure_rejects_a_window_it_cannot_draw(
    record: SignalTrace, zoom: tuple[float, float], message: str
) -> None:
    """An empty or reversed window would produce an axis with nothing on it."""
    with pytest.raises(ValueError, match=message):
        onset_marks_figure(record.times_s, record.signal, 1600, (), zoom_s=zoom)


def test_the_onset_figure_requires_matching_lengths(record: SignalTrace) -> None:
    """A signal on a different grid from its times is not a trace."""
    with pytest.raises(ValueError, match="same length"):
        onset_marks_figure(record.times_s, record.signal[:-1], 1600, (), zoom_s=(0.7, 1.0))


def test_the_signal_overview_figure_draws_a_panel_for_every_component(
    record: SignalTrace,
) -> None:
    """Signal, clean component, excitation and spectrum, in that order."""
    spectrum = welch_spectrum(record.signal, SAMPLE_RATE_HZ)
    figure = signal_overview_figure(record, spectrum)
    assert len(figure.axes) == 4
    assert figure.axes[3].get_yscale() == "log"
    assert spectrum.method in figure.axes[3].get_title()


def test_the_filter_response_figure_draws_one_curve_per_design() -> None:
    """Three designs give three gain curves and three group delay curves."""
    designs = (
        design_bandpass(SAMPLE_RATE_HZ),
        design_highpass(SAMPLE_RATE_HZ),
        design_powerline_notch(SAMPLE_RATE_HZ),
    )
    grid = np.linspace(1.0, 0.5 * SAMPLE_RATE_HZ - 1.0, 512)
    figure = filter_response_figure(designs, grid)
    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) == len(designs)
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels == [design.name for design in designs]


def test_the_fatigue_figure_shows_the_fitted_line_over_the_measured_points() -> None:
    """Two panels, and a trend line spanning the same epochs as the measurements."""
    trace = run_fatigue_protocol(FatigueSpec(duration_s=8.0, epoch_s=2.0))
    median_trend, _ = analyse_fatigue(trace)
    figure = fatigue_figure(trace, median_trend)
    assert len(figure.axes) == 2
    measured, fitted = figure.axes[0].lines[0], figure.axes[0].lines[1]
    assert list(measured.get_ydata()) == pytest.approx(list(trace.median_frequency_hz))
    assert list(fitted.get_xdata()) == pytest.approx(list(trace.epoch_start_s))
    assert f"{median_trend.slope_hz_per_s:.3f}" in str(fitted.get_label())
    assert f"{trace.median_frequency_hz[0]:.1f}" in figure.axes[0].get_title()


def test_the_detector_comparison_figure_draws_three_measures_per_detector() -> None:
    """Detection rate, false positive rate and bias, each against the same axis."""
    spec = SweepSpec(snr_db=(0.0, 20.0), n_trials=2)
    trace = run_detector_sweep((HodgesBuiDetector(), EnvelopeThresholdDetector()), spec)
    metrics = summarise_sweep(trace)
    figure = detector_comparison_figure(metrics)
    assert len(figure.axes) == 3
    labels = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert labels == {HodgesBuiDetector().name, EnvelopeThresholdDetector().name}


def test_saving_writes_a_png_and_creates_the_directory(tmp_path: Path) -> None:
    """The tracked figures live in a directory the script is allowed to create."""
    figure = latency_figure((_measurement("only", 30.0, 8.0),))
    target = tmp_path / "nested" / "figure.png"
    written = save(figure, target)
    assert written == target
    assert target.read_bytes()[:8] == PNG_MAGIC


def test_a_lower_resolution_save_produces_a_smaller_file(tmp_path: Path) -> None:
    """The size budget is met by resolution, so the override has to actually bite."""
    items = tuple(
        _measurement(name, latency, ripple)
        for name, latency, ripple in (("a", 20.0, 22.0), ("b", 60.0, 12.0), ("c", 110.0, 7.0))
    )
    default = save(latency_figure(items), tmp_path / "default.png")
    reduced = save(latency_figure(items), tmp_path / "reduced.png", dpi=90.0)
    assert reduced.stat().st_size < default.stat().st_size


def test_every_figure_function_returns_a_figure_with_an_attached_canvas() -> None:
    """The pyplot state machine is never used, so the canvas has to be explicit."""
    figure = latency_figure((_measurement("only", 30.0, 8.0),))
    assert isinstance(figure, Figure)
    assert figure.canvas is not None


def test_the_estimators_used_in_the_published_figure_are_all_measurable() -> None:
    """Every estimator drawn on the tracked figure reports a finite design delay."""
    published = (MovingAverageEnvelope(0.100), LowPassEnvelope(2.0), ExponentialEnvelope(0.05))
    for estimator in published:
        assert np.isfinite(estimator.nominal_delay_samples(SAMPLE_RATE_HZ))
