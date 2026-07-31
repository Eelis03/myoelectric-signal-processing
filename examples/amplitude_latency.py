"""Compare amplitude estimators for proportional control by smoothing against latency.

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
from myoelectric.analysis.reporting import format_latency_table
from myoelectric.pipeline.latency import LatencySpec, run_latency_study

SAMPLE_RATE_HZ = 2000.0

# Farrell and Weir (2007) report that the delay a myoelectric prosthesis user tolerates
# has an upper bound near this value, measured from the muscle contracting to the
# device responding. The amplitude estimator is only one contributor to that total.
CONTROLLER_DELAY_BUDGET_MS = 125.0


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
        # transient rather than the step response.
        estimators = (estimators[1], estimators[3], estimators[5], estimators[8])
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
    print(
        f"Every estimator listed stays inside the {CONTROLLER_DELAY_BUDGET_MS:.0f} ms "
        "delay budget on its own, but the budget also has to cover the onset decision "
        "delay, the classifier, and the actuator, so the slower settings leave little "
        "room."
    )

    if not arguments.no_figures:
        from myoelectric.analysis.figures import latency_figure, save

        path = save(
            latency_figure(trace.measurements), arguments.outdir / "amplitude_latency.png"
        )
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
