"""Compute the whole feature library on one contraction window and check its behaviour.

Run with:

    uv run python examples/feature_report.py
"""

from __future__ import annotations

import numpy as np
from _common import heading, parse_arguments

from myoelectric.algorithm.features_freq import frequency_domain_features, welch_spectrum
from myoelectric.algorithm.features_time import time_domain_features
from myoelectric.algorithm.filters import (
    apply_causal,
    cascade,
    design_bandpass,
    design_powerline_notch,
)
from myoelectric.analysis.reporting import (
    format_frequency_feature_table,
    format_time_feature_table,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

SAMPLE_RATE_HZ = 2000.0


def main() -> None:
    """Report the feature vector of one window and its behaviour under a change of scale."""
    arguments = parse_arguments(__doc__ or "")
    duration = 1.5 if arguments.quick else 3.0
    sampling = SamplingSpec(sample_rate_hz=SAMPLE_RATE_HZ, duration_s=duration)
    trace = generate(
        GenerationSpec(
            sampling=sampling,
            profile=ContractionProfile.single(0.3, duration - 0.2, 0.6, rise_s=0.1, fall_s=0.1),
            noise=NoiseSpec(snr_db=20.0),
            powerline=PowerlineSpec(),
        ),
        np.random.default_rng(20260731),
    )
    chain = cascade(
        (design_bandpass(SAMPLE_RATE_HZ), design_powerline_notch(SAMPLE_RATE_HZ)),
        name="bandpass then notch",
        rationale="Standard preprocessing before feature extraction.",
    )
    filtered = apply_causal(chain, trace.signal)

    rest = filtered[sampling.to_samples(0.05) : sampling.to_samples(0.25)]
    threshold = 3.0 * float(np.std(rest))
    slope_threshold = float(np.var(np.diff(rest)))
    window = filtered[sampling.to_samples(1.0) : sampling.to_samples(1.25)]

    heading("Window")
    print(
        f"length                     {window.size} samples ({window.size / SAMPLE_RATE_HZ:.3f} s)"
    )
    print(f"resting standard deviation {float(np.std(rest)):.4f}")
    print(f"amplitude threshold        {threshold:.4f} (three resting standard deviations)")
    print(
        f"slope threshold            {slope_threshold:.6f} "
        "(variance of the resting first difference, units of amplitude squared)"
    )

    features = time_domain_features(
        window, amplitude_threshold=threshold, slope_threshold=slope_threshold
    )
    heading("Time domain features")
    print(format_time_feature_table(features))

    spectrum = welch_spectrum(window, SAMPLE_RATE_HZ, segment_s=0.125).band(20.0, 450.0)
    spectral = frequency_domain_features(spectrum)
    heading("Frequency domain features")
    print(f"method: {spectral.method}")
    print()
    print(format_frequency_feature_table(spectral))

    heading("Behaviour under a change of scale")
    gain = 2.0
    scaled = time_domain_features(
        gain * window,
        amplitude_threshold=gain * threshold,
        slope_threshold=gain**2 * slope_threshold,
    )
    scaled_spectrum = welch_spectrum(gain * window, SAMPLE_RATE_HZ, segment_s=0.125).band(
        20.0, 450.0
    )
    scaled_spectral = frequency_domain_features(scaled_spectrum)
    rows = (
        ("mean absolute value", features.mean_absolute_value, scaled.mean_absolute_value, gain),
        ("root mean square", features.root_mean_square, scaled.root_mean_square, gain),
        ("waveform length", features.waveform_length, scaled.waveform_length, gain),
        ("integrated electromyogram", features.integrated_emg, scaled.integrated_emg, gain),
        ("variance", features.variance, scaled.variance, gain**2),
        ("zero crossings", features.zero_crossings, scaled.zero_crossings, 1.0),
        ("slope sign changes", features.slope_sign_changes, scaled.slope_sign_changes, 1.0),
        ("Willison amplitude", features.willison_amplitude, scaled.willison_amplitude, 1.0),
        (
            "median frequency",
            spectral.median_frequency_hz,
            scaled_spectral.median_frequency_hz,
            1.0,
        ),
        ("mean frequency", spectral.mean_frequency_hz, scaled_spectral.mean_frequency_hz, 1.0),
    )
    print(f"| Feature | Original | Scaled by {gain:g} | Ratio | Expected ratio |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, original, after, expected in rows:
        ratio = after / original if original else float("nan")
        print(f"| {name} | {original:.4f} | {after:.4f} | {ratio:.4f} | {expected:.4f} |")
    print(
        "\nAmplitude features scale with the gain, variance scales with its square, and "
        "the counting and spectral features are unchanged provided their thresholds are "
        "scaled with the signal."
    )


if __name__ == "__main__":
    main()
