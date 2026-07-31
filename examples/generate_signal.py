"""Generate one synthetic surface electromyogram and describe it.

Run with:

    uv run python examples/generate_signal.py
"""

from __future__ import annotations

import numpy as np
from _common import heading, parse_arguments

from myoelectric.algorithm.features_freq import (
    mean_frequency,
    median_frequency,
    welch_spectrum,
)
from myoelectric.algorithm.filters import (
    apply_causal,
    cascade,
    design_bandpass,
    design_powerline_notch,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPool, MotorUnitPoolSpec
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate
from myoelectric.pipeline.loaders import recording_from_trace


def main() -> None:
    """Generate a record, report its properties, and optionally draw it."""
    arguments = parse_arguments(__doc__ or "")
    duration = 1.5 if arguments.quick else 3.0
    sampling = SamplingSpec(sample_rate_hz=2000.0, duration_s=duration)
    profile = ContractionProfile.single(
        onset_s=0.5,
        offset_s=duration - 0.3,
        plateau_excitation=0.6,
        rise_s=0.15,
        fall_s=0.15,
        label="wrist flexion",
    )
    spec = GenerationSpec(
        sampling=sampling,
        profile=profile,
        pool=MotorUnitPoolSpec(),
        noise=NoiseSpec(snr_db=15.0),
        powerline=PowerlineSpec(fundamental_hz=50.0, n_harmonics=3),
        motion=MotionArtefactSpec(cutoff_hz=5.0, amplitude_ratio=0.5),
    )
    trace = generate(spec, np.random.default_rng(20260731))

    pool = MotorUnitPool.from_spec(spec.pool)
    heading("Motor unit pool")
    print(f"units                      {len(pool)}")
    print(
        f"recruitment thresholds     {pool.units[0].recruitment_threshold:.4f} "
        f"to {pool.units[-1].recruitment_threshold:.4f}"
    )
    print(
        f"action potential amplitude {pool.units[0].amplitude:.2f} "
        f"to {pool.units[-1].amplitude:.2f}"
    )
    print(
        f"time constants             {1e3 * pool.units[0].time_constant_s:.2f} ms "
        f"to {1e3 * pool.units[-1].time_constant_s:.2f} ms"
    )

    heading("Generated record")
    print(f"sample rate                {sampling.sample_rate_hz:.0f} Hz")
    print(f"duration                   {sampling.duration_s:.2f} s")
    print(f"neural onset               sample {trace.neural_onset_indices[0]}")
    print(f"first discharge            sample {trace.onset_indices[0]}")
    print(f"active clean rms           {trace.active_rms:.4f}")
    print(f"achieved signal to noise   {trace.achieved_snr_db:.2f} dB")
    print(f"rest rms before onset      {np.sqrt(np.mean(trace.clean[:1000] ** 2)):.3e}")

    chain = cascade(
        (design_bandpass(sampling.sample_rate_hz), design_powerline_notch(sampling.sample_rate_hz)),
        name="bandpass 20-450 Hz then 50 Hz notch cascade",
        rationale="Removes movement artefact, out of band noise, and mains interference.",
    )
    filtered = apply_causal(chain, trace.signal)
    active = slice(trace.onset_indices[0] + 400, sampling.n_samples - 600)

    raw_spectrum = welch_spectrum(trace.signal[active], sampling.sample_rate_hz)
    clean_spectrum = welch_spectrum(filtered[active], sampling.sample_rate_hz)
    heading("Spectral description of the active segment")
    print(f"method                     {clean_spectrum.method}")
    print(
        f"unfiltered median / mean   {median_frequency(raw_spectrum):.1f} Hz "
        f"/ {mean_frequency(raw_spectrum):.1f} Hz"
    )
    print(
        f"filtered median / mean     {median_frequency(clean_spectrum):.1f} Hz "
        f"/ {mean_frequency(clean_spectrum):.1f} Hz"
    )

    recording = recording_from_trace(trace, label=profile.events[0].label)
    heading("As an EmgRecording")
    print(f"channels                   {recording.n_channels} {recording.channel_names}")
    print(f"samples                    {recording.n_samples}")
    print(f"label                      {recording.label}")
    print(f"annotated onsets           {recording.onset_indices}")

    if not arguments.no_figures:
        from myoelectric.analysis.figures import save, signal_overview_figure

        path = save(
            signal_overview_figure(trace, clean_spectrum),
            arguments.outdir / "signal_overview.png",
        )
        print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
