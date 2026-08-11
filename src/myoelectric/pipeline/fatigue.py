"""Sustained contraction protocol for the muscle fatigue demonstration.

Localised muscular fatigue slows muscle fibre conduction velocity, which lengthens each
motor unit action potential in time and therefore compresses the power spectrum of the
surface signal towards lower frequencies. The median frequency of the spectrum falls
over a sustained contraction as a result. This is one of the most reliably reproduced
effects in surface electromyography, described by De Luca (1984) and quantified by
Merletti, Knaflitz and De Luca (1990), which is why it is used here as a validation of
the spectral feature implementation rather than as a novel result: an implementation
that fails to reproduce a downward median frequency trend is wrong.

The protocol generates a sustained contraction in epochs. Each epoch is generated with
a motor unit action potential duration scale that grows linearly from ``1.0`` at the
start of the contraction to ``duration_scale_end`` at the end. Because the Hermite
Rodriguez potential has a magnitude spectrum peaking at ``1 / (pi lambda)``, scaling
``lambda`` by ``s`` moves the peak by ``1 / s``, so an end scale of 1.35 corresponds to
a conduction velocity fall of about 26 per cent, which is within the range reported for
sustained contractions held to the limit of endurance.

Generating epoch by epoch means the potential duration is piecewise constant rather
than continuously varying. Since each epoch is analysed separately, and the spectral
estimate within an epoch sees a constant potential duration, the approximation does not
affect the quantity being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.features_freq import (
    mean_frequency,
    median_frequency,
    welch_spectrum,
)
from myoelectric.algorithm.features_time import root_mean_square
from myoelectric.algorithm.filters import FilterDesign, FilterMode, apply_filter
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPoolSpec
from myoelectric.model.noise import NoiseSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

__all__ = ["FatigueSpec", "FatigueTrace", "run_fatigue_protocol"]


@dataclass(frozen=True, slots=True)
class FatigueSpec:
    """Configuration of the sustained contraction protocol.

    Attributes:
        duration_s: Total length of the sustained contraction.
        epoch_s: Length of one analysis epoch. The spectral resolution of an epoch is
            ``sample_rate / segment_samples``, not ``1 / epoch_s``, because each epoch
            is itself split into Welch segments.
        sample_rate_hz: Sample rate.
        excitation: Constant normalised excitation held throughout.
        duration_scale_end: Motor unit action potential duration scale at the end of the
            contraction, relative to 1.0 at the start.
        snr_db: Signal to noise ratio of every epoch.
        band_hz: Band to which the spectrum is restricted before the features are taken.
        segment_s: Welch segment length inside an epoch.
        seed: Seed of the generator.
        pool: Motor unit pool.
        preprocess: Optional filter applied to each epoch before analysis.
        preprocess_mode: Mode in which that filter is applied. Zero phase is the correct
            choice here and is the default, because the analysis is offline, the whole
            record is available, and no timing decision depends on the result.
    """

    duration_s: float = 60.0
    epoch_s: float = 2.0
    sample_rate_hz: float = 2000.0
    excitation: float = 0.6
    duration_scale_end: float = 1.35
    snr_db: float = 20.0
    band_hz: tuple[float, float] = (20.0, 450.0)
    segment_s: float = 0.25
    seed: int = 20260731
    pool: MotorUnitPoolSpec = field(default_factory=MotorUnitPoolSpec)
    preprocess: FilterDesign | None = None
    preprocess_mode: FilterMode = "zero_phase"

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or self.epoch_s <= 0.0:
            raise ValueError("duration_s and epoch_s must be positive")
        if self.n_epochs < 3:
            raise ValueError("the protocol needs at least three epochs to fit a trend")
        if self.duration_scale_end <= 0.0:
            raise ValueError("duration_scale_end must be positive")

    @property
    def n_epochs(self) -> int:
        """Number of whole epochs in the contraction."""
        return int(self.duration_s // self.epoch_s)


@dataclass(frozen=True, slots=True, eq=False)
class FatigueTrace:
    """Per epoch spectral and amplitude features over a sustained contraction."""

    spec: FatigueSpec
    epoch_start_s: NDArray[np.float64]
    median_frequency_hz: NDArray[np.float64]
    mean_frequency_hz: NDArray[np.float64]
    root_mean_square: NDArray[np.float64]
    duration_scale: NDArray[np.float64]
    spectrum_method: str

    @property
    def n_epochs(self) -> int:
        """Number of epochs analysed."""
        return int(self.epoch_start_s.size)


def run_fatigue_protocol(spec: FatigueSpec) -> FatigueTrace:
    """Generate a sustained contraction and extract per epoch features."""
    rng = np.random.default_rng(spec.seed)
    sampling = SamplingSpec(sample_rate_hz=spec.sample_rate_hz, duration_s=spec.epoch_s)
    profile = ContractionProfile.single(
        onset_s=0.0,
        offset_s=spec.epoch_s,
        plateau_excitation=spec.excitation,
        rise_s=0.0,
        fall_s=0.0,
        label="sustained",
    )

    starts: list[float] = []
    medians: list[float] = []
    means: list[float] = []
    amplitudes: list[float] = []
    scales: list[float] = []
    method = ""

    last = max(1, spec.n_epochs - 1)
    for epoch in range(spec.n_epochs):
        scale = 1.0 + (spec.duration_scale_end - 1.0) * epoch / last
        trace = generate(
            GenerationSpec(
                sampling=sampling,
                profile=profile,
                pool=spec.pool,
                noise=NoiseSpec(snr_db=spec.snr_db),
                muap_duration_scale=scale,
            ),
            rng,
        )
        samples = trace.signal
        if spec.preprocess is not None:
            samples = apply_filter(spec.preprocess, samples, spec.preprocess_mode)
        spectrum = welch_spectrum(samples, spec.sample_rate_hz, segment_s=spec.segment_s).band(
            *spec.band_hz
        )
        method = spectrum.method
        starts.append(epoch * spec.epoch_s)
        medians.append(median_frequency(spectrum))
        means.append(mean_frequency(spectrum))
        amplitudes.append(root_mean_square(samples))
        scales.append(scale)

    return FatigueTrace(
        spec=spec,
        epoch_start_s=np.asarray(starts, dtype=np.float64),
        median_frequency_hz=np.asarray(medians, dtype=np.float64),
        mean_frequency_hz=np.asarray(means, dtype=np.float64),
        root_mean_square=np.asarray(amplitudes, dtype=np.float64),
        duration_scale=np.asarray(scales, dtype=np.float64),
        spectrum_method=method,
    )
