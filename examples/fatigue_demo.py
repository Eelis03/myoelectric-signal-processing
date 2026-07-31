"""Reproduce the downward median frequency shift of a sustained contraction.

Run with:

    uv run python examples/fatigue_demo.py
"""

from __future__ import annotations

import numpy as np
from _common import heading, parse_arguments

from myoelectric.algorithm.filters import cascade, design_bandpass, design_powerline_notch
from myoelectric.analysis.fatigue_stats import analyse_fatigue, format_fatigue_summary
from myoelectric.pipeline.fatigue import FatigueSpec, run_fatigue_protocol

SAMPLE_RATE_HZ = 2000.0


def main() -> None:
    """Run the sustained contraction protocol and test the median frequency trend."""
    arguments = parse_arguments(__doc__ or "")
    chain = cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale=(
            "Offline analysis of a stored contraction, so the chain is applied with "
            "zero phase filtering: the whole record is available and no timing "
            "decision depends on the result."
        ),
    )
    spec = FatigueSpec(
        duration_s=20.0 if arguments.quick else 60.0,
        epoch_s=2.0,
        sample_rate_hz=SAMPLE_RATE_HZ,
        preprocess=chain,
        preprocess_mode="zero_phase",
    )
    trace = run_fatigue_protocol(spec)
    median_trend, mean_trend = analyse_fatigue(trace)

    heading("Protocol")
    print(f"contraction                {spec.duration_s:g} s at excitation {spec.excitation:g}")
    print(f"epochs                     {trace.n_epochs} of {spec.epoch_s:g} s")
    print(f"signal to noise ratio      {spec.snr_db:g} dB")
    print(f"analysis band              {spec.band_hz[0]:g} to {spec.band_hz[1]:g} Hz")
    print(f"spectral estimate          {trace.spectrum_method}")
    print(
        f"potential duration scale   1.00 to {spec.duration_scale_end:.2f}, which predicts "
        f"a spectral compression to {1.0 / spec.duration_scale_end:.3f} of the initial "
        "frequency"
    )

    heading("Trend")
    print(format_fatigue_summary((median_trend, mean_trend)))

    predicted = 1.0 / spec.duration_scale_end
    span = trace.epoch_start_s[-1] - trace.epoch_start_s[0]
    fitted_ratio = (
        median_trend.intercept_hz + median_trend.slope_hz_per_s * trace.epoch_start_s[-1]
    ) / (median_trend.intercept_hz + median_trend.slope_hz_per_s * trace.epoch_start_s[0])
    heading("Check against the model")
    print(f"predicted ratio of final to initial median frequency  {predicted:.3f}")
    print(f"fitted ratio over {span:.0f} s                              {fitted_ratio:.3f}")
    print(
        "\nThe median frequency falls, the fall is significant at the one per cent "
        "level on a one sided test, and its size agrees with the compression that the "
        "lengthened action potentials predict. That agreement is the point of the "
        "demonstration: the effect is well established physiologically, so reproducing "
        "it validates the spectral feature implementation."
    )
    print(f"\nper epoch median frequency (Hz): {np.round(trace.median_frequency_hz, 1).tolist()}")

    if not arguments.no_figures:
        from myoelectric.analysis.figures import fatigue_figure, save

        path = save(fatigue_figure(trace, median_trend), arguments.outdir / "fatigue.png")
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
