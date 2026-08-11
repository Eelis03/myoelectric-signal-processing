"""Compare amplitude estimators for proportional control, and check a chain's budget.

Run with:

    uv run python examples/amplitude_latency.py
"""

from __future__ import annotations

from _common import heading, parse_arguments

from myoelectric.algorithm.envelope import (
    EnvelopeEstimator,
    ExponentialEnvelope,
    LowPassEnvelope,
    MovingAverageEnvelope,
    MovingRmsEnvelope,
)
from myoelectric.algorithm.filters import cascade, design_bandpass, design_powerline_notch
from myoelectric.algorithm.onset import HodgesBuiDetector
from myoelectric.analysis.delay_budget import (
    FARRELL_WEIR_LIMIT_MS,
    DelayBudgetExceededError,
    assemble_budget,
    detector_stage,
    enforce,
    envelope_stage,
    filter_stage,
    format_budget_table,
)
from myoelectric.analysis.reporting import format_latency_table
from myoelectric.pipeline.latency import LatencySpec, run_latency_study

SAMPLE_RATE_HZ = 2000.0

# The band over which the conditioning filter has to carry signal, and therefore the
# band over which its worst case group delay is charged to the budget.
ANALYSIS_BAND_HZ = (20.0, 450.0)


def main() -> None:
    """Measure the delay and the plateau ripple of every amplitude estimator."""
    arguments = parse_arguments(__doc__ or "")
    chain = cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Causal preprocessing, as a real time controller would run it.",
    )
    estimators: tuple[EnvelopeEstimator, ...] = (
        MovingAverageEnvelope(0.050),
        MovingAverageEnvelope(0.100),
        MovingAverageEnvelope(0.200),
        MovingRmsEnvelope(0.100),
        LowPassEnvelope(2.0),
        LowPassEnvelope(4.0),
        LowPassEnvelope(8.0),
        ExponentialEnvelope(0.025),
        ExponentialEnvelope(0.050),
    )
    if arguments.quick:
        # The record length is not reduced: a 2 Hz low pass needs about a second to
        # settle before the step, and shortening the record would measure the settling
        # transient rather than the step response. One estimator of each family is kept,
        # including the two the delay budget below quotes by name.
        estimators = (estimators[1], estimators[3], estimators[4], estimators[8])
    spec = LatencySpec(
        sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=3.0,
        step_s=1.0,
        preprocess=chain,
        preprocess_mode="causal",
    )
    trace = run_latency_study(estimators, spec)

    heading("Step contraction")
    print(f"sample rate                {spec.sample_rate_hz:.0f} Hz")
    print(f"step at                    {spec.step_s:g} s, zero rise time")
    print(f"plateau excitation         {spec.plateau_excitation:g}")
    print(f"achieved signal to noise   {trace.achieved_snr_db:.2f} dB")

    heading("Smoothing against latency")
    print(format_latency_table(trace.measurements))

    heading("Reading of the table")
    print(
        "Measured latency tracks the nominal group delay for the moving average and "
        "the low pass estimators, which is the check that the delay figures quoted in "
        "the design are the delays the estimators impose."
    )
    print(
        "The moving root mean square reaches half amplitude in a quarter of its window "
        "rather than half, because the mean square ramps linearly across the window and "
        "the square root compresses the first part of that ramp."
    )
    print(
        "The exponential estimator reaches half amplitude at 0.69 of its group delay, "
        "which is the natural logarithm of two, because its step response is a single "
        "exponential rather than a ramp."
    )

    heading("Delay budget of a whole real time chain")
    detector = HodgesBuiDetector()
    envelope = ExponentialEnvelope(0.050)
    conditioning = filter_stage(chain, ANALYSIS_BAND_HZ)
    stages = (
        conditioning,
        detector_stage(detector, SAMPLE_RATE_HZ),
        envelope_stage(envelope, SAMPLE_RATE_HZ),
    )
    budget = assemble_budget(stages)
    print(
        "Conditioning, onset decision and amplitude estimation, each charged the delay "
        "it imposes. Every estimator above stays inside the budget on its own; the "
        "question is what the chain costs once they are added together."
    )
    print()
    print(format_budget_table(budget))
    enforce(budget)
    print()
    print(
        f"The three stages this library implements consume {budget.total_ms:.1f} ms of "
        f"the {FARRELL_WEIR_LIMIT_MS:.0f} ms bound, leaving {budget.headroom_ms:.1f} ms "
        "for the classifier and the actuator, which this library does not implement and "
        "which therefore do not appear as rows above. The largest single contributor is "
        f"{budget.dominant_stage.name} at {budget.dominant_stage.delay_ms:.1f} ms."
    )

    heading("The same chain with the smoothest amplitude estimator")
    smoothest = LowPassEnvelope(2.0)
    over = assemble_budget((*stages[:2], envelope_stage(smoothest, SAMPLE_RATE_HZ)))
    print(format_budget_table(over))
    print()
    try:
        enforce(over)
    except DelayBudgetExceededError as error:
        print(f"DelayBudgetExceededError: {error}")
    else:  # pragma: no cover - the chain is over budget by construction
        raise AssertionError("the smoothest estimator was expected to exceed the budget")
    ripple = {item.estimator: item.plateau_ripple_percent for item in trace.measurements}
    print(
        f"\nThat is the check the library previously did not make. {smoothest.name} is "
        f"the steadiest estimator in the table above at {ripple[smoothest.name]:.1f} per "
        f"cent ripple, against {ripple[envelope.name]:.1f} per cent for {envelope.name}, "
        "and choosing it on that basis alone puts the chain over the budget by "
        f"{-over.headroom_ms:.1f} ms."
    )

    heading("The same conditioning charged as a strict bound instead")
    strict = filter_stage(chain, ANALYSIS_BAND_HZ, rule="worst_case")
    print(f"power weighted   {conditioning.delay_ms:7.2f} ms   {conditioning.basis}")
    print(f"worst case       {strict.delay_ms:7.2f} ms   {strict.basis}")
    print(
        "\nThe two differ by a factor of twenty and the whole difference is the notch. "
        "Group delay peaks where a magnitude response falls fastest, so the largest "
        "figure in the band belongs to a component the notch exists to destroy. The "
        "power weighted rule gives that component the weight its surviving amplitude "
        "earns, which is almost none, and is the delay a broadband envelope actually "
        "experiences. The worst case rule is reported alongside because a bound that "
        "is never quoted is a bound nobody checked."
    )

    heading("Why a zero phase stage cannot be entered into the budget")
    try:
        filter_stage(chain, ANALYSIS_BAND_HZ, mode="zero_phase")
    except ValueError as error:
        print(f"ValueError: {error}")
    else:  # pragma: no cover - zero phase is rejected by construction
        raise AssertionError("a zero phase stage was expected to be refused")

    if not arguments.no_figures:
        from myoelectric.analysis.figures import latency_figure, save

        path = save(latency_figure(trace.measurements), arguments.outdir / "amplitude_latency.png")
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
