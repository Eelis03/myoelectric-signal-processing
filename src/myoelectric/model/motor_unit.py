"""Motor unit pool with size ordered recruitment and rate coding.

The pool follows the organisation described by Fuglevand, Winter and Patla (1993).
Recruitment thresholds are distributed exponentially across the pool, discharge rate
rises linearly with excitation above threshold and saturates at a unit specific peak,
and motor unit action potential amplitude grows with recruitment threshold so that the
units recruited last contribute the largest potentials.

The action potential shape is the second order Hermite Rodriguez function

    h(u) = (1 - 2 u^2) exp(-u^2),    u = t / lambda

which is the negative second derivative of a Gaussian up to a constant. Hermite
Rodriguez functions are the standard compact support basis for modelling motor unit
action potentials (Lo Conte, Merletti and Sandri, 1994). Two properties matter here.
First, the function integrates to zero over the real line, so a simulated potential
adds no direct current offset. Second, its magnitude spectrum is proportional to
``w^2 exp(-w^2 lambda^2 / 4)``, which peaks at ``f = 1 / (pi lambda)``. Choosing
``lambda`` between 2.8 ms and 4.5 ms therefore places the spectral peak of individual
potentials between about 71 Hz and 114 Hz, which is the range reported for surface
recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["MotorUnit", "MotorUnitPool", "MotorUnitPoolSpec"]

# The Hermite Rodriguez function is truncated at this many time constants either side
# of its centre. At u = 3 the envelope exp(-u^2) is 1.2e-4 of its peak, so the
# truncation error is far below the amplitude resolution of any real amplifier.
_SUPPORT_TIME_CONSTANTS: float = 3.0


@dataclass(frozen=True, slots=True)
class MotorUnitPoolSpec:
    """Parameters of a motor unit pool.

    Attributes:
        n_units: Number of motor units in the pool.
        recruitment_range: Ratio of the highest to the lowest recruitment threshold.
            Fuglevand et al. (1993) use 30 for a pool spanning the usual physiological
            range, meaning the last unit is recruited at thirty times the excitation
            that recruits the first.
        min_firing_rate_hz: Discharge rate of a unit at the instant it is recruited.
        peak_firing_rate_hz: Peak discharge rate of the first recruited unit.
        peak_firing_spread_hz: Amount by which the peak discharge rate falls from the
            first to the last recruited unit.
        firing_rate_gain_hz: Increase in discharge rate per unit of normalised
            excitation above threshold.
        interspike_interval_cv: Coefficient of variation of the interspike interval.
            Values near 0.2 match the discharge variability of human motor units.
        amplitude_range: Ratio of the largest to the smallest action potential
            amplitude across the pool.
        time_constant_first_s: Hermite Rodriguez time constant of the first recruited
            unit, which is the slowest and therefore the lowest in frequency.
        time_constant_last_s: Hermite Rodriguez time constant of the last recruited
            unit, which is the fastest.
    """

    n_units: int = 24
    recruitment_range: float = 30.0
    min_firing_rate_hz: float = 8.0
    peak_firing_rate_hz: float = 35.0
    peak_firing_spread_hz: float = 10.0
    firing_rate_gain_hz: float = 30.0
    interspike_interval_cv: float = 0.2
    amplitude_range: float = 40.0
    time_constant_first_s: float = 4.5e-3
    time_constant_last_s: float = 2.8e-3

    def __post_init__(self) -> None:
        if self.n_units < 2:
            raise ValueError(f"n_units must be at least 2, got {self.n_units}")
        if self.recruitment_range <= 1.0:
            raise ValueError("recruitment_range must exceed 1")
        if self.amplitude_range <= 0.0:
            raise ValueError("amplitude_range must be positive")
        if self.min_firing_rate_hz <= 0.0:
            raise ValueError("min_firing_rate_hz must be positive")
        if self.peak_firing_rate_hz <= self.min_firing_rate_hz:
            raise ValueError("peak_firing_rate_hz must exceed min_firing_rate_hz")
        if self.peak_firing_spread_hz < 0.0:
            raise ValueError("peak_firing_spread_hz must not be negative")
        if self.interspike_interval_cv < 0.0:
            raise ValueError("interspike_interval_cv must not be negative")
        if min(self.time_constant_first_s, self.time_constant_last_s) <= 0.0:
            raise ValueError("motor unit time constants must be positive")


@dataclass(frozen=True, slots=True)
class MotorUnit:
    """One motor unit: when it is recruited, how fast it fires, and what it looks like."""

    index: int
    recruitment_threshold: float
    min_firing_rate_hz: float
    peak_firing_rate_hz: float
    firing_rate_gain_hz: float
    amplitude: float
    time_constant_s: float

    def firing_rate_hz(self, excitation: float) -> float:
        """Discharge rate at a normalised excitation in ``[0, 1]``.

        Returns zero below the recruitment threshold. Above it the rate rises linearly
        from ``min_firing_rate_hz`` with slope ``firing_rate_gain_hz`` and saturates at
        ``peak_firing_rate_hz``.
        """
        if excitation < self.recruitment_threshold:
            return 0.0
        rate = self.min_firing_rate_hz + self.firing_rate_gain_hz * (
            excitation - self.recruitment_threshold
        )
        return min(rate, self.peak_firing_rate_hz)

    def action_potential(
        self, sample_rate_hz: float, duration_scale: float = 1.0
    ) -> NDArray[np.float64]:
        """Sampled action potential, centred, scaled to unit peak magnitude times amplitude.

        Args:
            sample_rate_hz: Sample rate of the record the potential will be placed in.
            duration_scale: Multiplier on the time constant. Values above one widen the
                potential in time and therefore compress its spectrum towards lower
                frequencies, which is how slowed muscle fibre conduction velocity is
                represented during a fatiguing contraction.
        """
        if duration_scale <= 0.0:
            raise ValueError("duration_scale must be positive")
        time_constant = self.time_constant_s * duration_scale
        half_width = int(np.ceil(_SUPPORT_TIME_CONSTANTS * time_constant * sample_rate_hz))
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        scaled = offsets / (time_constant * sample_rate_hz)
        shape = (1.0 - 2.0 * scaled**2) * np.exp(-(scaled**2))
        # Remove the residual offset introduced by truncating an otherwise zero mean
        # function, so that a train of potentials carries no direct current component.
        shape = shape - float(np.mean(shape))
        peak = float(np.max(np.abs(shape)))
        return np.asarray(self.amplitude * shape / peak, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MotorUnitPool:
    """An ordered collection of motor units built from a :class:`MotorUnitPoolSpec`."""

    spec: MotorUnitPoolSpec
    units: tuple[MotorUnit, ...] = field(default=())

    @classmethod
    def from_spec(cls, spec: MotorUnitPoolSpec) -> MotorUnitPool:
        """Build the pool described by ``spec``.

        Recruitment thresholds follow ``exp(ln(R) * i / (n - 1)) / R`` for
        ``i = 0 ... n - 1``, which spreads them exponentially over ``[1 / R, 1]`` with
        many low threshold units and few high threshold units, as in Fuglevand et al.
        (1993). Amplitudes follow the same exponential law over ``[1, A]`` so that
        amplitude increases with recruitment threshold, which is the size principle.
        """
        last = spec.n_units - 1
        positions = np.arange(spec.n_units, dtype=np.float64) / last
        thresholds = np.exp(np.log(spec.recruitment_range) * positions) / spec.recruitment_range
        amplitudes = np.exp(np.log(spec.amplitude_range) * positions)
        peak_rates = spec.peak_firing_rate_hz - spec.peak_firing_spread_hz * positions
        time_constants = np.exp(
            np.log(spec.time_constant_first_s)
            + positions * np.log(spec.time_constant_last_s / spec.time_constant_first_s)
        )
        units = tuple(
            MotorUnit(
                index=i,
                recruitment_threshold=float(thresholds[i]),
                min_firing_rate_hz=spec.min_firing_rate_hz,
                peak_firing_rate_hz=float(peak_rates[i]),
                firing_rate_gain_hz=spec.firing_rate_gain_hz,
                amplitude=float(amplitudes[i]),
                time_constant_s=float(time_constants[i]),
            )
            for i in range(spec.n_units)
        )
        return cls(spec=spec, units=units)

    @property
    def lowest_threshold(self) -> float:
        """Recruitment threshold of the first unit to be recruited."""
        return self.units[0].recruitment_threshold

    def __len__(self) -> int:
        return len(self.units)
