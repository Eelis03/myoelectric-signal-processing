"""Specifications for the three contaminations added to a synthetic recording.

Three contaminations dominate practical surface electromyography: wideband noise from
the instrumentation and the tissue, power line interference picked up by the electrode
leads, and low frequency movement artefact produced by electrode and cable motion.
Each is specified separately so that a filter can be judged against the contamination
it was designed to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MotionArtefactSpec", "NoiseSpec", "PowerlineSpec"]


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Additive wideband Gaussian noise, specified by signal to noise ratio.

    The ratio is defined as ``20 log10(rms_active_signal / rms_noise)`` where the
    numerator is the root mean square of the noise free motor unit potential train
    restricted to the samples where at least one unit is active. Power line
    interference and movement artefact are excluded from this ratio and are specified
    separately, because they are deterministic or narrowband contaminations rather
    than the noise floor.
    """

    snr_db: float = 10.0


@dataclass(frozen=True, slots=True)
class PowerlineSpec:
    """Mains interference at a fundamental frequency and its harmonics.

    Attributes:
        fundamental_hz: Mains frequency, 50 Hz in Europe and 60 Hz in North America.
        n_harmonics: Number of components including the fundamental. Components above
            the Nyquist frequency are discarded.
        amplitude_ratio: Amplitude of the fundamental relative to the root mean square
            of the active clean signal.
        harmonic_decay: Multiplier applied to the amplitude of each successive
            harmonic, so component ``k`` has amplitude
            ``amplitude_ratio * harmonic_decay ** k``.
    """

    fundamental_hz: float = 50.0
    n_harmonics: int = 3
    amplitude_ratio: float = 0.30
    harmonic_decay: float = 0.5

    def __post_init__(self) -> None:
        if self.fundamental_hz <= 0.0:
            raise ValueError("fundamental_hz must be positive")
        if self.n_harmonics < 1:
            raise ValueError("n_harmonics must be at least 1")
        if self.amplitude_ratio < 0.0:
            raise ValueError("amplitude_ratio must not be negative")
        if not 0.0 <= self.harmonic_decay <= 1.0:
            raise ValueError("harmonic_decay must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class MotionArtefactSpec:
    """Low frequency movement artefact modelled as low pass filtered Gaussian noise.

    Attributes:
        cutoff_hz: Corner of the low pass shaping filter. Movement artefact energy in
            surface recordings is concentrated below about 20 Hz.
        amplitude_ratio: Root mean square of the artefact relative to the root mean
            square of the active clean signal.
    """

    cutoff_hz: float = 5.0
    amplitude_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive")
        if self.amplitude_ratio < 0.0:
            raise ValueError("amplitude_ratio must not be negative")
