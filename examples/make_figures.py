"""Regenerate the three figures that are tracked in the repository.

Run with:

    uv run python examples/make_figures.py

Three figures are published rather than every figure the examples can draw, because a
figure earns its place only if it shows something the tables cannot. The fatigue trend
shows a physiological effect being recovered from a spectrum. The onset panel shows why
one detector is late and another is early, which a column of biases states but does not
explain. The latency frontier shows which amplitude estimators are dominated, which is
not readable off nine rows of numbers.

Size is a deliberate constraint. The three files together are held under 250 KB, which
is met by writing at a resolution matched to the width a README is read at, and by
reducing dense traces to one vertical extent per pixel column before drawing them. No
compression dependency is used; every saving comes from drawing less.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import TRACKED_FIGURES, heading, parse_arguments

from myoelectric.algorithm.envelope import (
    EnvelopeEstimator,
    ExponentialEnvelope,
    LowPassEnvelope,
    MovingAverageEnvelope,
    MovingRmsEnvelope,
)
from myoelectric.algorithm.filters import (
    apply_causal,
    cascade,
    design_bandpass,
    design_powerline_notch,
)
from myoelectric.algorithm.onset import (
    BonatoDetector,
    EnvelopeThresholdDetector,
    HodgesBuiDetector,
)
from myoelectric.analysis.fatigue_stats import analyse_fatigue
from myoelectric.analysis.figures import (
    fatigue_figure,
    latency_figure,
    onset_marks_figure,
    pareto_frontier,
    save,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.fatigue import FatigueSpec, run_fatigue_protocol
from myoelectric.pipeline.generation import GenerationSpec, generate
from myoelectric.pipeline.latency import LatencySpec, run_latency_study

SAMPLE_RATE_HZ = 2000.0
SEED = 20260731

# A figure displayed at README width is about 800 pixels across. Ninety dots per inch
# over a nine inch figure gives 810, so nothing is discarded at the size the figure is
# actually viewed at, and nothing is paid for beyond it.
PUBLISHED_DPI = 90.0

# Total size budget for the tracked figures, checked after writing so that a change
# which quietly inflates them fails here rather than in the portfolio validator.
BUDGET_BYTES = 250 * 1024

ESTIMATORS: tuple[EnvelopeEstimator, ...] = (
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


def _fatigue(outdir: Path, quick: bool) -> Path:
    """Median frequency over a sustained contraction, with the fitted trend."""
    chain = cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Offline analysis, so the chain is applied with zero phase filtering.",
    )
    spec = FatigueSpec(
        duration_s=20.0 if quick else 60.0,
        epoch_s=2.0,
        sample_rate_hz=SAMPLE_RATE_HZ,
        preprocess=chain,
        preprocess_mode="zero_phase",
    )
    trace = run_fatigue_protocol(spec)
    median_trend, _ = analyse_fatigue(trace)
    print(
        f"  median frequency {trace.median_frequency_hz[0]:.1f} Hz to "
        f"{trace.median_frequency_hz[-1]:.1f} Hz over {spec.duration_s:g} s, "
        f"slope {median_trend.slope_hz_per_s:.3f} Hz/s"
    )
    return save(
        fatigue_figure(trace, median_trend),
        outdir / "fatigue-median-frequency.png",
        dpi=PUBLISHED_DPI,
    )


def _onsets(outdir: Path, quick: bool) -> Path:
    """One conditioned record with the ground truth and all three detector answers."""
    del quick  # One two second record either way; there is nothing to reduce.
    sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=2.0)
    trace = generate(
        GenerationSpec(
            sampling=sampling,
            profile=ContractionProfile.single(
                onset_s=0.8, offset_s=1.8, plateau_excitation=0.5, rise_s=0.02, fall_s=0.1
            ),
            noise=NoiseSpec(snr_db=5.0),
            powerline=PowerlineSpec(),
            motion=MotionArtefactSpec(),
        ),
        np.random.default_rng(SEED),
    )
    conditioned = apply_causal(design_bandpass(SAMPLE_RATE_HZ), trace.signal)
    detectors = (EnvelopeThresholdDetector(), HodgesBuiDetector(), BonatoDetector())
    results = tuple(detector.detect(conditioned, SAMPLE_RATE_HZ) for detector in detectors)
    truth = trace.onset_indices[0]
    print(f"  ground truth first discharge at sample {truth}, {truth / SAMPLE_RATE_HZ:.3f} s")
    for result in results:
        first = result.first_onset_index
        error = (
            "not detected" if first is None else f"{1e3 * (first - truth) / SAMPLE_RATE_HZ:+.0f} ms"
        )
        print(f"  {result.detector:24s} {error}")
    return save(
        onset_marks_figure(trace.times_s, conditioned, truth, results, zoom_s=(0.72, 0.92)),
        outdir / "onset-detectors.png",
        dpi=PUBLISHED_DPI,
    )


def _latency(outdir: Path, quick: bool) -> Path:
    """Plateau ripple against measured latency, with the non dominated frontier."""
    del quick  # The record cannot be shortened: a 2 Hz low pass needs a second to settle.
    chain = cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Causal preprocessing, as a real time controller would run it.",
    )
    trace = run_latency_study(
        ESTIMATORS,
        LatencySpec(
            sample_rate_hz=SAMPLE_RATE_HZ,
            duration_s=3.0,
            step_s=1.0,
            preprocess=chain,
            preprocess_mode="causal",
        ),
    )
    frontier = pareto_frontier(trace.measurements)
    dominated = tuple(
        item.estimator
        for item in trace.measurements
        if item.estimator not in {row.estimator for row in frontier}
    )
    print(f"  non dominated: {', '.join(item.estimator for item in frontier)}")
    print(f"  dominated:     {', '.join(dominated) if dominated else 'none'}")
    return save(
        latency_figure(trace.measurements),
        outdir / "amplitude-latency-ripple.png",
        dpi=PUBLISHED_DPI,
    )


def main() -> None:
    """Write every tracked figure and report the size against the budget."""
    arguments = parse_arguments(__doc__ or "", default_outdir=TRACKED_FIGURES)
    outdir = Path(arguments.outdir)

    heading("Tracked figures")
    if arguments.no_figures:
        print(f"nothing written: --no-figures was given, target was {outdir}")
        return

    written: list[Path] = []
    for label, build in (
        ("fatigue-median-frequency.png", _fatigue),
        ("onset-detectors.png", _onsets),
        ("amplitude-latency-ripple.png", _latency),
    ):
        print(f"\n{label}")
        written.append(build(outdir, arguments.quick))

    heading("Size")
    total = 0
    for path in written:
        size = path.stat().st_size
        total += size
        print(f"{path.name:34s} {size / 1024:7.1f} KB")
    print(f"{'total':34s} {total / 1024:7.1f} KB of a {BUDGET_BYTES / 1024:.0f} KB budget")
    if total > BUDGET_BYTES:
        raise SystemExit(
            f"tracked figures total {total} bytes, over the {BUDGET_BYTES} byte budget"
        )


if __name__ == "__main__":
    main()
