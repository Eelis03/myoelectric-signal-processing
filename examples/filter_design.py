"""Report the designed filters, their group delays, and the cost of zero phase filtering.

Run with:

    uv run python examples/filter_design.py
"""

from __future__ import annotations

import numpy as np
from _common import heading, parse_arguments

from myoelectric.algorithm.filters import (
    apply_causal,
    apply_zero_phase,
    cascade,
    design_bandpass,
    design_highpass,
    design_powerline_notch,
)
from myoelectric.analysis.reporting import format_filter_response_table
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

SAMPLE_RATE_HZ = 2000.0


def _envelope_peak_index(x: np.ndarray) -> float:
    """Centre of mass of the squared signal, a robust estimate of a burst position."""
    weight = np.asarray(x, dtype=np.float64) ** 2
    total = float(np.sum(weight))
    return float(np.sum(np.arange(weight.size) * weight) / total) if total > 0.0 else float("nan")


def main() -> None:
    """Print the filter tables and measure the delay each application mode imposes."""
    arguments = parse_arguments(__doc__ or "")
    bandpass = design_bandpass(SAMPLE_RATE_HZ)
    notch = design_powerline_notch(SAMPLE_RATE_HZ)
    highpass = design_highpass(SAMPLE_RATE_HZ)

    for design in (bandpass, notch, highpass):
        heading(design.name)
        print(design.rationale)
        print(f"sections {design.n_sections}, poles {design.order}")

    heading("Band pass response")
    print(
        format_filter_response_table(
            bandpass, (5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 450.0, 600.0, 800.0)
        )
    )

    heading("Power line notch response, causal")
    print(
        format_filter_response_table(
            notch, (45.0, 48.0, 50.0, 52.0, 55.0, 100.0, 150.0), modes=("causal",)
        )
    )
    print()
    print(
        "Group delay is undefined at a notch centre because the response is zero "
        "there, which is why those rows read nan."
    )

    heading("Movement artefact high pass response, causal")
    print(format_filter_response_table(highpass, (2.0, 5.0, 10.0, 20.0, 50.0, 200.0), ("causal",)))

    heading("Measured delay of an amplitude modulated burst at 100 Hz")
    sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=1.0)
    times = sampling.times()
    burst = np.exp(-(((times - 0.5) / 0.05) ** 2)) * np.sin(2.0 * np.pi * 100.0 * times)
    causal = apply_causal(bandpass, burst)
    zero_phase = apply_zero_phase(bandpass, burst)
    reference = _envelope_peak_index(burst)
    predicted = float(bandpass.group_delay_samples(np.array([100.0]), mode="causal")[0])
    print(f"predicted group delay at 100 Hz  {predicted:.2f} samples")
    print(
        f"measured shift, causal           "
        f"{_envelope_peak_index(causal) - reference:.2f} samples"
    )
    print(
        f"measured shift, zero phase       "
        f"{_envelope_peak_index(zero_phase) - reference:.2f} samples"
    )
    print(
        "\nThe causal shift matches the design group delay. The zero phase shift is "
        "zero because the reverse pass cancels the forward phase, which is only "
        "possible when the samples after the burst are already available."
    )

    heading("Contamination removal on a generated record")
    duration = 1.0 if arguments.quick else 2.0
    record_sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=duration)
    trace = generate(
        GenerationSpec(
            sampling=record_sampling,
            profile=ContractionProfile.single(0.2, duration - 0.1, 0.6, rise_s=0.05, fall_s=0.05),
            noise=NoiseSpec(snr_db=15.0),
            powerline=PowerlineSpec(fundamental_hz=50.0, n_harmonics=3),
            motion=MotionArtefactSpec(cutoff_hz=5.0, amplitude_ratio=0.5),
        ),
        np.random.default_rng(20260731),
    )
    chain = cascade(
        (bandpass, notch),
        name="bandpass then notch",
        rationale="Band limit first, then remove the mains components that remain in band.",
    )
    filtered = apply_causal(chain, trace.signal)
    grid = np.array([49.0, 50.0, 51.0, 100.0, 150.0], dtype=np.float64)
    gains = chain.gain_db(grid)
    print("chain gain near the mains components")
    for frequency, gain in zip(grid, gains, strict=True):
        print(f"  {frequency:6.1f} Hz  {gain:8.2f} dB")
    print(
        "\nThe gain at exactly 50 Hz is a transmission zero, so the printed value is "
        "limited only by floating point. What matters in practice is what the chain "
        "does to each contamination, measured below over the settled part of the "
        "record. The first 0.5 s is excluded because a causal recursive filter starts "
        "from a zero state and needs time to settle, and the settling transient is not "
        "part of the steady state attenuation."
    )
    settled = slice(record_sampling.to_samples(0.5), record_sampling.n_samples)
    for label, component in (
        ("power line", trace.powerline),
        ("movement artefact", trace.motion_artefact),
        ("wideband noise", trace.noise),
        ("clean signal", trace.clean),
    ):
        before = float(np.sqrt(np.mean(component[settled] ** 2)))
        after = float(np.sqrt(np.mean(apply_causal(chain, component)[settled] ** 2)))
        ratio = 20.0 * np.log10(after / before) if before > 0.0 else float("nan")
        print(f"  {label:20s} rms {before:.4f} -> {after:.4f}  ({ratio:+.1f} dB)")
    print(f"\nrms before filtering             {np.sqrt(np.mean(trace.signal**2)):.4f}")
    print(f"rms after filtering              {np.sqrt(np.mean(filtered**2)):.4f}")
    print(f"rms of the clean component       {np.sqrt(np.mean(trace.clean**2)):.4f}")

    if not arguments.no_figures:
        from myoelectric.analysis.figures import filter_response_figure, save

        grid = np.linspace(1.0, 0.5 * SAMPLE_RATE_HZ - 1.0, 2000)
        path = save(
            filter_response_figure((bandpass, highpass, notch), grid),
            arguments.outdir / "filter_response.png",
        )
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
