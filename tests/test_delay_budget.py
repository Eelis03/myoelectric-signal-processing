"""Delay budget accounting.

The properties tested here are the ones that make a summed delay mean something: that
the sum is the sum, that a bound is above the value it bounds, that a frequency
dependent group delay is reduced to one number by a rule the stage records, that a
non causal stage cannot enter a controller budget at all, and that a chain over the
limit raises rather than being reported and ignored.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from myoelectric.algorithm.envelope import (
    ExponentialEnvelope,
    LowPassEnvelope,
    MovingAverageEnvelope,
)
from myoelectric.algorithm.filters import (
    FilterDesign,
    cascade,
    design_bandpass,
    design_lowpass,
    design_powerline_notch,
)
from myoelectric.algorithm.onset import BonatoDetector, HodgesBuiDetector
from myoelectric.analysis.delay_budget import (
    FARRELL_WEIR_LIMIT_MS,
    DelayBudget,
    DelayBudgetExceededError,
    DelayStage,
    assemble_budget,
    detector_stage,
    enforce,
    envelope_stage,
    filter_stage,
    fixed_stage,
    format_budget_table,
)
from tests.helpers import SAMPLE_RATE_HZ


@pytest.fixture
def conditioning_chain() -> FilterDesign:
    """Band pass followed by the mains notch, the chain a controller would run."""
    return cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Causal conditioning.",
    )


def test_a_stage_cannot_carry_a_negative_delay() -> None:
    """A stage that advanced the signal in time would be non causal."""
    with pytest.raises(ValueError, match="negative delay"):
        DelayStage(name="impossible", delay_ms=-1.0, basis="test")


def test_a_stage_cannot_carry_a_delay_that_is_not_finite() -> None:
    """A group delay evaluated at a transmission zero must not reach a budget."""
    with pytest.raises(ValueError, match="not finite"):
        DelayStage(name="undefined", delay_ms=float("nan"), basis="test")


def test_the_total_is_the_sum_of_the_stages() -> None:
    """Group delays add in cascade, so the total is a plain sum with no correction."""
    stages = (
        fixed_stage("a", 10.0, "given"),
        fixed_stage("b", 2.5, "given"),
        fixed_stage("c", 0.25, "given"),
    )
    budget = assemble_budget(stages, limit_ms=100.0)
    assert budget.total_ms == pytest.approx(12.75)
    assert budget.headroom_ms == pytest.approx(87.25)
    assert budget.within_budget
    assert budget.dominant_stage.name == "a"


def test_a_budget_needs_at_least_one_stage() -> None:
    """An empty budget would report a total of zero, which is not a measurement."""
    with pytest.raises(ValueError, match="at least one stage"):
        assemble_budget(())


def test_a_budget_needs_a_positive_limit() -> None:
    """A limit of zero or less could never be met."""
    with pytest.raises(ValueError, match="limit_ms must be positive"):
        DelayBudget(stages=(fixed_stage("a", 1.0, "given"),), limit_ms=0.0)


def test_the_budget_is_met_exactly_at_the_limit() -> None:
    """The comparison is inclusive, so a chain landing exactly on the limit passes."""
    budget = assemble_budget((fixed_stage("a", 50.0, "given"),), limit_ms=50.0)
    assert budget.within_budget
    assert budget.headroom_ms == pytest.approx(0.0)
    assert enforce(budget) is budget


def test_enforce_raises_and_names_the_largest_contributor() -> None:
    """The exception has to say where the delay went, or it cannot be acted on."""
    budget = assemble_budget(
        (fixed_stage("small", 10.0, "given"), fixed_stage("large", 200.0, "given")),
        limit_ms=125.0,
    )
    assert not budget.within_budget
    with pytest.raises(DelayBudgetExceededError, match="large"):
        enforce(budget)


def test_a_detector_stage_carries_its_decision_delay(sample_rate_hz: float) -> None:
    """Hodges and Bui must observe its whole 25 ms window before forming the mean."""
    detector = HodgesBuiDetector(window_s=0.025)
    stage = detector_stage(detector, sample_rate_hz)
    assert stage.delay_ms == pytest.approx(25.0)
    assert stage.name == detector.name


def test_a_bonato_stage_is_the_cheapest_of_the_three(sample_rate_hz: float) -> None:
    """It tests the signal itself rather than a smoothed envelope, so it waits least."""
    delays = [
        detector_stage(detector, sample_rate_hz).delay_ms
        for detector in (HodgesBuiDetector(), BonatoDetector())
    ]
    assert delays[1] < delays[0]


def test_an_envelope_stage_matches_the_closed_form_kernel_delay(sample_rate_hz: float) -> None:
    """A rectangular kernel of ``n`` samples has group delay ``(n - 1) / 2`` samples."""
    estimator = MovingAverageEnvelope(0.100)
    window = round(0.100 * sample_rate_hz)
    expected_ms = 1e3 * 0.5 * (window - 1) / sample_rate_hz
    assert envelope_stage(estimator, sample_rate_hz).delay_ms == pytest.approx(expected_ms)


def test_an_exponential_envelope_stage_matches_its_recursion(sample_rate_hz: float) -> None:
    """The single pole recursion has group delay ``(1 - a) / a`` samples at zero."""
    estimator = ExponentialEnvelope(0.050)
    alpha = estimator.alpha(sample_rate_hz)
    expected_ms = 1e3 * ((1.0 - alpha) / alpha) / sample_rate_hz
    assert envelope_stage(estimator, sample_rate_hz).delay_ms == pytest.approx(expected_ms)


def test_a_zero_phase_stage_is_refused(conditioning_chain: FilterDesign) -> None:
    """Its zero group delay is bought by reading samples that do not exist yet."""
    with pytest.raises(ValueError, match="reads samples that have not been acquired"):
        filter_stage(conditioning_chain, (20.0, 450.0), mode="zero_phase")


def test_both_rules_agree_where_the_group_delay_is_flat(sample_rate_hz: float) -> None:
    """Over a narrow slice of a flat pass band there is only one delay to report.

    Band pass group delay varies by less than a tenth of a millisecond between 200 Hz
    and 300 Hz, so the weighted mean, the worst case, and the value at the midpoint all
    have to agree to within that variation. The tolerance is the variation itself,
    measured from the design, not a number chosen to make the test pass.
    """
    design = design_bandpass(sample_rate_hz)
    grid = np.linspace(200.0, 300.0, 512)
    variation = float(np.max(design.group_delay_ms(grid)) - np.min(design.group_delay_ms(grid)))
    midpoint = float(design.group_delay_ms(np.array([250.0]))[0])

    weighted = filter_stage(design, (200.0, 300.0), rule="power_weighted").delay_ms
    worst = filter_stage(design, (200.0, 300.0), rule="worst_case").delay_ms
    assert weighted == pytest.approx(midpoint, abs=variation)
    assert worst == pytest.approx(midpoint, abs=variation)


def test_the_worst_case_rule_bounds_the_weighted_rule(sample_rate_hz: float) -> None:
    """A bound that a weighted average could exceed would not be a bound."""
    design = design_bandpass(sample_rate_hz)
    band = (20.0, 450.0)
    assert (
        filter_stage(design, band, rule="worst_case").delay_ms
        >= filter_stage(design, band, rule="power_weighted").delay_ms
    )


def test_the_band_pass_worst_case_is_reached_at_its_lower_corner(sample_rate_hz: float) -> None:
    """Butterworth group delay peaks at the corner, which is the expensive end."""
    stage = filter_stage(design_bandpass(sample_rate_hz), (20.0, 450.0), rule="worst_case")
    assert "20.0" in stage.basis or "20.2" in stage.basis
    assert stage.delay_ms > 25.0


def test_a_notch_costs_far_more_as_a_bound_than_as_a_weighted_mean(
    conditioning_chain: FilterDesign,
) -> None:
    """Group delay peaks where magnitude falls fastest, which is where signal is lost.

    The worst case in the pass band belongs to a component about one Hertz from the
    mains line, which the notch exists to destroy. Weighting by the squared magnitude
    gives that component the weight its surviving amplitude earns, and the two rules
    then differ by more than an order of magnitude. Both are reported by the library
    for exactly this reason, and the frequency at which the bound is reached is
    recorded on the stage so that the difference can be explained rather than argued
    about.
    """
    band = (20.0, 450.0)
    weighted = filter_stage(conditioning_chain, band, rule="power_weighted")
    worst = filter_stage(conditioning_chain, band, rule="worst_case")
    assert worst.delay_ms > 10.0 * weighted.delay_ms

    reached_hz = float(worst.basis.rsplit("reached at ", 1)[1].removesuffix(" Hz"))
    mains = design_powerline_notch(SAMPLE_RATE_HZ)
    assert abs(reached_hz - 50.0) < 2.0
    assert float(mains.gain_db(np.array([reached_hz]))[0]) < -1.0


def test_a_transmission_zero_alone_leaves_no_delay_to_report(sample_rate_hz: float) -> None:
    """A band containing only the notch centre has no defined group delay at all."""
    notch = design_powerline_notch(sample_rate_hz)
    with pytest.raises(ValueError, match="no defined group delay"):
        filter_stage(notch, (50.0 - 1e-9, 50.0 + 1e-9), rule="worst_case", n_points=3)


def test_the_band_must_lie_inside_the_sampled_spectrum(sample_rate_hz: float) -> None:
    """A band reaching past the Nyquist frequency describes nothing the filter sees."""
    design = design_bandpass(sample_rate_hz)
    with pytest.raises(ValueError, match="band_hz must satisfy"):
        filter_stage(design, (20.0, sample_rate_hz))
    with pytest.raises(ValueError, match="band_hz must satisfy"):
        filter_stage(design, (300.0, 100.0))


def test_the_grid_must_have_at_least_two_points(sample_rate_hz: float) -> None:
    """One point is a frequency, not a band."""
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        filter_stage(design_bandpass(sample_rate_hz), (20.0, 450.0), n_points=1)


def test_an_unknown_rule_is_refused(sample_rate_hz: float) -> None:
    """Silently defaulting would report a number under conditions nobody chose."""
    with pytest.raises(ValueError, match="unknown rule"):
        filter_stage(
            design_bandpass(sample_rate_hz),
            (20.0, 450.0),
            rule="average",  # type: ignore[arg-type]
        )


def test_a_stage_records_the_band_and_the_rule_it_was_computed_under(
    sample_rate_hz: float,
) -> None:
    """A delay without its conditions cannot be checked, so both go on the stage."""
    weighted = filter_stage(design_lowpass(sample_rate_hz, 8.0), (0.0, 40.0))
    worst = filter_stage(design_lowpass(sample_rate_hz, 8.0), (0.0, 40.0), rule="worst_case")
    assert "0 to 40 Hz" in weighted.basis
    assert "squared magnitude" in weighted.basis
    assert "0 to 40 Hz" in worst.basis
    assert "peak gain" in worst.basis


def test_a_realistic_chain_fits_and_names_its_largest_contributor(
    conditioning_chain: FilterDesign,
    sample_rate_hz: float,
) -> None:
    """Conditioning, detection and amplitude estimation fit inside the reported bound.

    The point of the check is not that this particular chain passes. It is that the
    total is now compared against something, so a chain that does not pass is caught.
    """
    budget = assemble_budget(
        (
            filter_stage(conditioning_chain, (20.0, 450.0)),
            detector_stage(HodgesBuiDetector(), sample_rate_hz),
            envelope_stage(ExponentialEnvelope(0.050), sample_rate_hz),
        )
    )
    assert budget.limit_ms == FARRELL_WEIR_LIMIT_MS
    assert budget.within_budget
    assert budget.dominant_stage.name == ExponentialEnvelope(0.050).name
    assert budget.total_ms == pytest.approx(sum(s.delay_ms for s in budget.stages))


def test_swapping_in_the_smoothest_estimator_breaks_the_budget(
    conditioning_chain: FilterDesign,
    sample_rate_hz: float,
) -> None:
    """The chain that minimises ripple is the chain that misses the delay bound."""
    stages = (
        filter_stage(conditioning_chain, (20.0, 450.0)),
        detector_stage(HodgesBuiDetector(), sample_rate_hz),
        envelope_stage(LowPassEnvelope(2.0), sample_rate_hz),
    )
    budget = assemble_budget(stages)
    assert not budget.within_budget
    assert budget.headroom_ms < 0.0
    with pytest.raises(DelayBudgetExceededError, match="lowpass-2Hz"):
        enforce(budget)


def test_the_table_lists_every_stage_the_total_and_the_verdict() -> None:
    """A budget that is not readable as a table cannot be put in a document."""
    budget = assemble_budget(
        (fixed_stage("conditioning", 4.0, "given"), fixed_stage("estimator", 50.0, "given")),
        limit_ms=125.0,
    )
    table = format_budget_table(budget)
    lines = table.splitlines()
    assert lines[0].startswith("| Stage |")
    assert len(lines) == 2 + len(budget.stages) + 2
    assert "conditioning" in table
    assert "estimator" in table
    assert "54.00" in table
    assert "within budget" in table
    assert "+71.00" in table


def test_the_table_says_so_when_the_chain_is_over_budget() -> None:
    """The verdict has to change with the verdict, not only the number."""
    budget = assemble_budget((fixed_stage("slow", 400.0, "given"),), limit_ms=125.0)
    table = format_budget_table(budget)
    assert "over budget" in table
    assert "-275.00" in table


def test_a_weighted_delay_is_never_below_the_smallest_group_delay_in_the_band(
    sample_rate_hz: float,
) -> None:
    """A weighted mean lies between the extremes of what it averages."""
    design = design_bandpass(sample_rate_hz)
    band = (20.0, 450.0)
    grid = np.linspace(band[0], band[1], 2048)
    delays = design.group_delay_ms(grid)
    weighted = filter_stage(design, band, rule="power_weighted").delay_ms
    assert float(np.nanmin(delays)) <= weighted <= float(np.nanmax(delays))
    assert math.isfinite(weighted)
