"""Compare three onset detectors across signal to noise ratio and across threshold.

Run with:

    uv run python examples/onset_benchmark.py
"""

from __future__ import annotations

from _common import heading, parse_arguments

from myoelectric.algorithm.filters import design_bandpass
from myoelectric.algorithm.onset import (
    BonatoDetector,
    EnvelopeThresholdDetector,
    HodgesBuiDetector,
)
from myoelectric.analysis.detector_metrics import format_metrics_table, summarise_sweep
from myoelectric.model.noise import PowerlineSpec
from myoelectric.pipeline.detection_sweep import SweepSpec, run_detector_sweep

SAMPLE_RATE_HZ = 2000.0


def main() -> None:
    """Run the signal to noise sweep and the threshold sensitivity study."""
    arguments = parse_arguments(__doc__ or "")
    quick = arguments.quick
    preprocess = design_bandpass(SAMPLE_RATE_HZ)

    detectors = (
        EnvelopeThresholdDetector(),
        HodgesBuiDetector(),
        BonatoDetector(),
    )
    sweep_spec = SweepSpec(
        snr_db=(0.0, 10.0, 20.0) if quick else (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
        n_trials=8 if quick else 60,
        powerline=PowerlineSpec(),
        preprocess=preprocess,
        preprocess_mode="causal",
    )
    trace = run_detector_sweep(detectors, sweep_spec)
    metrics = summarise_sweep(trace)

    heading("Onset detection against signal to noise ratio")
    print(
        f"{sweep_spec.n_trials} active trials and {sweep_spec.n_trials} resting trials "
        f"per ratio, {sweep_spec.sampling.duration_s:g} s each, "
        f"match tolerance {1e3 * sweep_spec.match_tolerance_s:.0f} ms, "
        f"causal band pass before detection."
    )
    print()
    print(format_metrics_table(metrics))
    print()
    print(
        "Every threshold is estimated from the resting baseline of the record being "
        "tested, so the false positive rate is a property of the threshold and the "
        "decision rule and does not vary with the amplitude of the contraction. With "
        f"{sweep_spec.n_trials} resting trials, a false positive rate reported as zero "
        f"has an upper 95 per cent bound of {3.0 / sweep_spec.n_trials:.3f} by the rule "
        "of three."
    )

    sensitivity = (
        EnvelopeThresholdDetector(threshold_sd=1.0),
        EnvelopeThresholdDetector(threshold_sd=2.0),
        EnvelopeThresholdDetector(threshold_sd=3.0),
        HodgesBuiDetector(threshold_sd=1.0),
        HodgesBuiDetector(threshold_sd=2.0),
        HodgesBuiDetector(threshold_sd=3.0),
        BonatoDetector(false_alarm_probability=1e-1),
        BonatoDetector(false_alarm_probability=1e-2),
        BonatoDetector(false_alarm_probability=1e-3),
    )
    sensitivity_spec = SweepSpec(
        snr_db=(5.0,),
        n_trials=8 if quick else 60,
        powerline=PowerlineSpec(),
        preprocess=preprocess,
        preprocess_mode="causal",
    )
    sensitivity_trace = run_detector_sweep(sensitivity, sensitivity_spec)
    sensitivity_metrics = summarise_sweep(sensitivity_trace)

    heading("Threshold sensitivity at 5 dB")
    print(
        "The same experiment repeated with each detector at three sensitivities. This "
        "is the operating characteristic that a single detection rate hides."
    )
    print()
    print(format_metrics_table(sensitivity_metrics))

    heading("Decision delay of each detector at 2000 Hz")
    for detector in detectors:
        print(f"{detector.name:28s} {1e3 * detector.decision_delay_s(SAMPLE_RATE_HZ):6.1f} ms")
    print(
        "\nThe timing bias above says where a detector places the onset. The decision "
        "delay says how long after that instant the detector can say so. A controller "
        "pays the sum."
    )

    if not arguments.no_figures:
        from myoelectric.analysis.figures import detector_comparison_figure, save

        path = save(
            detector_comparison_figure(metrics), arguments.outdir / "detector_comparison.png"
        )
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
