"""Onset detector evaluation swept over signal to noise ratio.

Each trial produces two records from the same generator settings and the same noise
level: an active record containing one contraction with a known ground truth onset, and
a resting record of the same length containing no contraction at all. The active record
measures whether the detector finds the onset and where it places it. The resting record
measures how often the detector declares an onset when there is nothing to find. A
detection rate reported without the matching false positive rate says nothing, because
a detector that fires constantly reaches a detection rate of one.

The two records of a trial use independent noise realisations, drawn from the same
seeded generator, so the pair is reproducible from the sweep seed alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from myoelectric.algorithm.filters import FilterDesign, FilterMode, apply_filter
from myoelectric.algorithm.onset import OnsetDetector
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPoolSpec
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate

__all__ = ["DetectorSweepTrace", "SweepSpec", "TrialOutcome", "run_detector_sweep"]


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """Configuration of one detector comparison.

    Attributes:
        snr_db: Signal to noise ratios to sweep, in decibels.
        n_trials: Number of independent trials at each ratio. The standard error of an
            estimated rate is at most ``0.5 / sqrt(n_trials)``, which is what fixes the
            tolerance used when the aggregate rates are pinned in a regression test.
        seed: Seed of the sweep generator.
        sampling: Sampling geometry of every record.
        onset_s: Neural onset time of the contraction in the active record.
        offset_s: Offset time of that contraction.
        plateau_excitation: Plateau excitation of that contraction.
        rise_s: Duration of the excitation ramp. The default of 20 ms represents a rapid
            contraction, which is the condition under which onset detectors are usually
            characterised. A longer ramp inflates every detector's timing bias by the
            time the muscle takes to reach a detectable amplitude, which is a property
            of the contraction rather than of the detector.
        match_tolerance_s: A detection counts as correct when it falls within this many
            seconds of the ground truth onset. The window is one sided in neither
            direction: detections before the onset are also counted as matches, so that
            a detector cannot gain credit by firing early on noise.
        pool: Motor unit pool used by the generator.
        powerline: Optional mains interference added to every record.
        motion: Optional movement artefact added to every record.
        preprocess: Optional filter applied to every record before detection.
        preprocess_mode: Mode in which that filter is applied. The default is causal,
            because a detector that is evaluated after a zero phase filter is not the
            detector that a controller would run.
    """

    snr_db: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    n_trials: int = 60
    seed: int = 20260731
    sampling: SamplingSpec = field(
        default_factory=lambda: SamplingSpec(sample_rate_hz=2000.0, duration_s=2.0)
    )
    onset_s: float = 0.8
    offset_s: float = 1.8
    plateau_excitation: float = 0.5
    rise_s: float = 0.02
    match_tolerance_s: float = 0.15
    pool: MotorUnitPoolSpec = field(default_factory=MotorUnitPoolSpec)
    powerline: PowerlineSpec | None = None
    motion: MotionArtefactSpec | None = None
    preprocess: FilterDesign | None = None
    preprocess_mode: FilterMode = "causal"

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if not self.snr_db:
            raise ValueError("snr_db must list at least one value")
        if self.match_tolerance_s <= 0.0:
            raise ValueError("match_tolerance_s must be positive")


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Result of one detector on one trial at one signal to noise ratio."""

    detector: str
    snr_db: float
    trial: int
    true_index: int
    detected_index: int | None
    matched: bool
    error_s: float
    rest_detections: int

    @property
    def error_ms(self) -> float:
        """Timing error in milliseconds, positive when the detection is late."""
        return 1e3 * self.error_s


@dataclass(frozen=True, slots=True, eq=False)
class DetectorSweepTrace:
    """Structured record of a whole sweep."""

    spec: SweepSpec
    detector_names: tuple[str, ...]
    outcomes: tuple[TrialOutcome, ...]
    rest_duration_s: float

    def for_detector(self, name: str) -> tuple[TrialOutcome, ...]:
        """Every outcome belonging to one detector."""
        return tuple(outcome for outcome in self.outcomes if outcome.detector == name)


def _make_spec(spec: SweepSpec, snr_db: float, profile: ContractionProfile) -> GenerationSpec:
    return GenerationSpec(
        sampling=spec.sampling,
        profile=profile,
        pool=spec.pool,
        noise=NoiseSpec(snr_db=snr_db),
        powerline=spec.powerline,
        motion=spec.motion,
    )


def run_detector_sweep(detectors: tuple[OnsetDetector, ...], spec: SweepSpec) -> DetectorSweepTrace:
    """Run every detector over every trial at every signal to noise ratio."""
    if not detectors:
        raise ValueError("at least one detector is required")

    rng = np.random.default_rng(spec.seed)
    active_profile = ContractionProfile.single(
        onset_s=spec.onset_s,
        offset_s=spec.offset_s,
        plateau_excitation=spec.plateau_excitation,
        rise_s=spec.rise_s,
    )
    rest_profile = ContractionProfile.rest_only()
    tolerance = spec.sampling.to_samples(spec.match_tolerance_s)
    sample_rate = spec.sampling.sample_rate_hz

    outcomes: list[TrialOutcome] = []
    for snr_db in spec.snr_db:
        for trial in range(spec.n_trials):
            active = generate(_make_spec(spec, snr_db, active_profile), rng)
            rest = generate(_make_spec(spec, snr_db, rest_profile), rng)
            true_index = active.first_onset_index
            if true_index is None:
                raise RuntimeError("the active record produced no ground truth onset")

            active_signal = active.signal
            rest_signal = rest.signal
            if spec.preprocess is not None:
                active_signal = apply_filter(spec.preprocess, active_signal, spec.preprocess_mode)
                rest_signal = apply_filter(spec.preprocess, rest_signal, spec.preprocess_mode)

            for detector in detectors:
                found = detector.detect(active_signal, sample_rate)
                rest_result = detector.detect(rest_signal, sample_rate)
                candidates = [
                    index for index in found.onset_indices if abs(index - true_index) <= tolerance
                ]
                detected = candidates[0] if candidates else found.first_onset_index
                matched = bool(candidates)
                error = (
                    spec.sampling.to_seconds(detected - true_index)
                    if matched and detected is not None
                    else float("nan")
                )
                outcomes.append(
                    TrialOutcome(
                        detector=detector.name,
                        snr_db=snr_db,
                        trial=trial,
                        true_index=true_index,
                        detected_index=detected,
                        matched=matched,
                        error_s=error,
                        rest_detections=rest_result.n_detections,
                    )
                )

    return DetectorSweepTrace(
        spec=spec,
        detector_names=tuple(detector.name for detector in detectors),
        outcomes=tuple(outcomes),
        rest_duration_s=spec.sampling.duration_s,
    )
