"""Synthetic surface electromyogram generation.

No dataset is downloaded or committed by this project, so every number it reports comes
from a signal generated here. The generator is documented in full so that a reader can
judge what the reported numbers do and do not establish. See
``docs/design-notes.md`` for that discussion and :mod:`myoelectric.pipeline.loaders`
for the interface through which a real recording can be substituted.

Generation proceeds in five stages.

1. Excitation. A :class:`~myoelectric.model.contraction.ContractionProfile` gives a
   normalised neural excitation for every sample.

2. Discharge times. Each motor unit in the pool fires as an inhomogeneous renewal
   process. Starting from the first time its recruitment threshold is exceeded, the
   next discharge follows after an interval drawn as ``(1 / rate) (1 + cv z)`` where
   ``rate`` is the discharge rate at the current excitation, ``z`` is standard normal
   and ``cv`` is the interspike interval coefficient of variation. Intervals are
   clipped at a refractory floor so that a unit cannot fire twice in the same
   millisecond. This is the rate coding scheme of Fuglevand, Winter and Patla (1993).

3. Potential trains. The discharge train of each unit is convolved with that unit's
   action potential and the results are summed. The convolution is performed once for
   the whole pool as a batched fast Fourier transform, which is exact up to floating
   point rounding and much faster than looping over units.

4. Scaling. The summed train is scaled so that its root mean square over the samples in
   which at least one unit is active equals ``target_rms``. Fixing the active root mean
   square is what makes the signal to noise ratio of the next stage meaningful.

5. Contamination. Wideband Gaussian noise is added at the requested signal to noise
   ratio, followed by optional power line interference and optional movement artefact,
   each scaled relative to the same active root mean square.

Ground truth. The true onset of muscle activity is the time of the first motor unit
discharge at or after the neural onset of the event. This is earlier than any detector
can report, because a single discharge does not by itself raise the signal above the
noise, and it is later than the neural onset, because excitation must first climb to
the lowest recruitment threshold. Both are recorded in the trace so that a reported
timing bias can be interpreted against the definition that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.motor_unit import MotorUnitPool, MotorUnitPoolSpec
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec

__all__ = ["GenerationSpec", "SignalTrace", "generate"]

# A motor unit cannot discharge twice within this interval, which bounds the sampled
# interspike interval from below and keeps the renewal process well defined even if a
# large negative variate is drawn.
_REFRACTORY_S: float = 5e-3


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Everything needed to generate one synthetic record."""

    sampling: SamplingSpec
    profile: ContractionProfile
    pool: MotorUnitPoolSpec = field(default_factory=MotorUnitPoolSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    powerline: PowerlineSpec | None = None
    motion: MotionArtefactSpec | None = None
    target_rms: float = 1.0
    muap_duration_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.target_rms <= 0.0:
            raise ValueError("target_rms must be positive")
        if self.muap_duration_scale <= 0.0:
            raise ValueError("muap_duration_scale must be positive")


@dataclass(frozen=True, slots=True, eq=False)
class SignalTrace:
    """A generated record together with every component that went into it."""

    sampling: SamplingSpec
    signal: NDArray[np.float64]
    clean: NDArray[np.float64]
    noise: NDArray[np.float64]
    powerline: NDArray[np.float64]
    motion_artefact: NDArray[np.float64]
    excitation: NDArray[np.float64]
    onset_indices: tuple[int, ...]
    neural_onset_indices: tuple[int, ...]
    active_rms: float
    achieved_snr_db: float

    @property
    def times_s(self) -> NDArray[np.float64]:
        """Sample times in seconds."""
        return self.sampling.times()

    @property
    def first_onset_index(self) -> int | None:
        """Ground truth index of the first onset, or ``None`` for a resting record."""
        return self.onset_indices[0] if self.onset_indices else None


def _discharge_times(
    pool: MotorUnitPool,
    excitation: NDArray[np.float64],
    sampling: SamplingSpec,
    rng: np.random.Generator,
) -> list[NDArray[np.int64]]:
    """Sample indices at which each motor unit discharges.

    A newly recruited unit does not fire at the instant it is recruited. Its first
    discharge is placed uniformly at random within one interspike interval of that
    instant, which is the standard way to avoid an artificial synchronised volley at
    the start of a contraction. Without it, every unit whose threshold is exceeded by
    an abrupt rise in excitation would discharge on exactly the same sample and produce
    a transient that no real muscle generates and that flatters every onset detector.
    """
    times = sampling.times()
    duration = float(times[-1])
    trains: list[NDArray[np.int64]] = []
    for unit in pool.units:
        indices: list[int] = []
        active = excitation >= unit.recruitment_threshold
        if not bool(np.any(active)):
            trains.append(np.empty(0, dtype=np.int64))
            continue
        recruited_at = int(np.argmax(active))
        first_rate = unit.firing_rate_hz(float(excitation[recruited_at]))
        current = float(times[recruited_at]) + float(rng.uniform()) / max(first_rate, 1e-6)
        while current <= duration:
            index = round(float(current) * float(sampling.sample_rate_hz))
            if index >= sampling.n_samples:
                break
            rate = unit.firing_rate_hz(float(excitation[index]))
            if rate <= 0.0:
                remaining = np.flatnonzero(active[index:])
                if remaining.size == 0:
                    break
                restart = index + int(remaining[0])
                restart_rate = unit.firing_rate_hz(float(excitation[restart]))
                current = float(times[restart]) + float(rng.uniform()) / max(restart_rate, 1e-6)
                continue
            indices.append(index)
            interval = (1.0 / rate) * (
                1.0 + pool.spec.interspike_interval_cv * float(rng.standard_normal())
            )
            current += max(_REFRACTORY_S, interval)
        trains.append(np.asarray(indices, dtype=np.int64))
    return trains


def _potential_matrix(
    pool: MotorUnitPool, sampling: SamplingSpec, duration_scale: float
) -> NDArray[np.float64]:
    """Action potentials of every unit, right padded to a common length.

    Each potential is placed so that its first sample coincides with the discharge, not
    so that its centre does. The convolution below is therefore causal and the first
    non zero sample of the generated record is exactly the first discharge, which is
    what makes the ground truth onset unambiguous.
    """
    potentials = [
        unit.action_potential(sampling.sample_rate_hz, duration_scale) for unit in pool.units
    ]
    width = max(potential.size for potential in potentials)
    matrix = np.zeros((len(potentials), width), dtype=np.float64)
    for row, potential in enumerate(potentials):
        matrix[row, : potential.size] = potential
    return matrix


def _powerline_component(
    spec: PowerlineSpec,
    sampling: SamplingSpec,
    active_rms: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    times = sampling.times()
    component = np.zeros(sampling.n_samples, dtype=np.float64)
    for harmonic in range(spec.n_harmonics):
        frequency = spec.fundamental_hz * (harmonic + 1)
        if frequency >= sampling.nyquist_hz:
            break
        amplitude = active_rms * spec.amplitude_ratio * spec.harmonic_decay**harmonic
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        component += amplitude * np.sin(2.0 * np.pi * frequency * times + phase)
    return component


def _motion_component(
    spec: MotionArtefactSpec,
    sampling: SamplingSpec,
    active_rms: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    white = rng.standard_normal(sampling.n_samples)
    sos = sp_signal.butter(
        2, spec.cutoff_hz, btype="lowpass", output="sos", fs=sampling.sample_rate_hz
    )
    # The artefact is part of the signal model rather than part of the processing, so
    # shaping it with a zero phase filter is legitimate here: nothing downstream reads
    # its phase, and a causal shaping filter would leave a startup transient that the
    # artefact does not have in reality.
    shaped = np.asarray(sp_signal.sosfiltfilt(sos, white), dtype=np.float64)
    shaped_rms = float(np.sqrt(np.mean(shaped**2)))
    if shaped_rms <= 0.0:
        return np.zeros(sampling.n_samples, dtype=np.float64)
    return np.asarray(shaped * (active_rms * spec.amplitude_ratio / shaped_rms))


def generate(spec: GenerationSpec, rng: np.random.Generator) -> SignalTrace:
    """Generate one synthetic surface electromyogram record.

    Args:
        spec: The generation parameters.
        rng: A seeded generator. Passing ``numpy.random.default_rng(seed)`` makes the
            record reproducible: the PCG64 bit generator produces the same stream on
            every platform numpy supports.
    """
    sampling = spec.sampling
    pool = MotorUnitPool.from_spec(spec.pool)
    excitation = spec.profile.excitation(sampling.times())

    trains = _discharge_times(pool, excitation, sampling, rng)
    impulse = np.zeros((len(pool), sampling.n_samples), dtype=np.float64)
    for row, indices in enumerate(trains):
        if indices.size:
            np.add.at(impulse[row], indices, 1.0)

    potentials = _potential_matrix(pool, sampling, spec.muap_duration_scale)
    summed = sp_signal.fftconvolve(impulse, potentials, mode="full", axes=1)
    clean = np.asarray(np.sum(summed[:, : sampling.n_samples], axis=0), dtype=np.float64)

    active = excitation >= pool.lowest_threshold
    reference = clean[active] if bool(np.any(active)) else clean
    raw_rms = float(np.sqrt(np.mean(reference**2))) if reference.size else 0.0
    if raw_rms > 0.0:
        clean = clean * (spec.target_rms / raw_rms)
    active_rms = spec.target_rms if raw_rms > 0.0 else 0.0

    # A resting record has no active samples, so the noise level cannot be referred to
    # the signal. It is referred to target_rms instead, which keeps the noise floor of a
    # resting record identical to that of an active record at the same requested ratio.
    noise_reference = active_rms if active_rms > 0.0 else spec.target_rms
    noise_rms = noise_reference * 10.0 ** (-spec.noise.snr_db / 20.0)
    noise = np.asarray(rng.standard_normal(sampling.n_samples) * noise_rms, dtype=np.float64)

    powerline = (
        _powerline_component(spec.powerline, sampling, noise_reference, rng)
        if spec.powerline is not None
        else np.zeros(sampling.n_samples, dtype=np.float64)
    )
    motion = (
        _motion_component(spec.motion, sampling, noise_reference, rng)
        if spec.motion is not None
        else np.zeros(sampling.n_samples, dtype=np.float64)
    )

    signal = clean + noise + powerline + motion
    realised_noise_rms = float(np.sqrt(np.mean(noise**2)))
    achieved = (
        20.0 * float(np.log10(active_rms / realised_noise_rms))
        if active_rms > 0.0 and realised_noise_rms > 0.0
        else float("-inf")
    )

    onset_indices: list[int] = []
    neural_indices: list[int] = []
    populated = [train for train in trains if train.size]
    first_discharge = (
        np.sort(np.concatenate(populated)) if populated else np.empty(0, dtype=np.int64)
    )
    for onset_time in spec.profile.onset_times_s:
        neural_index = sampling.to_samples(onset_time)
        neural_indices.append(neural_index)
        after = first_discharge[first_discharge >= neural_index]
        if after.size:
            onset_indices.append(int(after[0]))

    return SignalTrace(
        sampling=sampling,
        signal=signal,
        clean=clean,
        noise=noise,
        powerline=powerline,
        motion_artefact=motion,
        excitation=excitation,
        onset_indices=tuple(onset_indices),
        neural_onset_indices=tuple(neural_indices),
        active_rms=active_rms,
        achieved_snr_db=achieved,
    )
