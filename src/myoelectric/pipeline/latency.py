"""Smoothing against latency for proportional control amplitude estimation.

The study generates one record containing a step change from rest to a steady
contraction, runs every amplitude estimator over it, and measures for each the delay it
imposes on the step and the residual ripple it leaves on the plateau. Those two numbers
are the trade off: any estimator can be made smoother by lengthening its window, and
every such change costs delay.

The step is created with zero rise time so that the input transition is instantaneous
and the measured delay belongs entirely to the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from myoelectric.algorithm.envelope import EnvelopeEstimator, LatencyMeasurement, measure_latency
from myoelectric.algorithm.filters import FilterDesign, FilterMode, apply_filter
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPoolSpec
from myoelectric.model.noise import NoiseSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

__all__ = ["LatencySpec", "LatencyTrace", "run_latency_study"]


@dataclass(frozen=True, slots=True)
class LatencySpec:
    """Configuration of the amplitude estimator comparison."""

    sample_rate_hz: float = 2000.0
    duration_s: float = 3.0
    step_s: float = 1.0
    plateau_excitation: float = 0.6
    snr_db: float = 20.0
    settle_s: float = 0.5
    seed: int = 20260731
    pool: MotorUnitPoolSpec = field(default_factory=MotorUnitPoolSpec)
    preprocess: FilterDesign | None = None
    preprocess_mode: FilterMode = "causal"

    def __post_init__(self) -> None:
        if not 0.0 < self.step_s < self.duration_s:
            raise ValueError("step_s must lie strictly inside the record")
        if self.settle_s <= 0.0:
            raise ValueError("settle_s must be positive")
        if self.step_s + self.settle_s >= self.duration_s:
            raise ValueError("the plateau must contain at least one settled sample")


@dataclass(frozen=True, slots=True)
class LatencyTrace:
    """Measured latency and ripple for every estimator in the study."""

    spec: LatencySpec
    measurements: tuple[LatencyMeasurement, ...]
    achieved_snr_db: float


def run_latency_study(estimators: tuple[EnvelopeEstimator, ...], spec: LatencySpec) -> LatencyTrace:
    """Measure the delay and ripple of every estimator on one step contraction."""
    if not estimators:
        raise ValueError("at least one estimator is required")
    rng = np.random.default_rng(spec.seed)
    sampling = SamplingSpec(sample_rate_hz=spec.sample_rate_hz, duration_s=spec.duration_s)
    profile = ContractionProfile.single(
        onset_s=spec.step_s,
        offset_s=spec.duration_s,
        plateau_excitation=spec.plateau_excitation,
        rise_s=0.0,
        fall_s=0.0,
        label="step",
    )
    trace = generate(
        GenerationSpec(
            sampling=sampling,
            profile=profile,
            pool=spec.pool,
            noise=NoiseSpec(snr_db=spec.snr_db),
        ),
        rng,
    )
    samples = trace.signal
    if spec.preprocess is not None:
        samples = apply_filter(spec.preprocess, samples, spec.preprocess_mode)

    step_index = sampling.to_samples(spec.step_s)
    plateau = slice(sampling.to_samples(spec.step_s + spec.settle_s), sampling.n_samples)
    measurements = tuple(
        measure_latency(estimator, samples, spec.sample_rate_hz, step_index, plateau)
        for estimator in estimators
    )
    return LatencyTrace(spec=spec, measurements=measurements, achieved_snr_db=trace.achieved_snr_db)
