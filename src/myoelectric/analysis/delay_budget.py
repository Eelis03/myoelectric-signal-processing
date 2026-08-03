"""Accounting for the delay a processing chain imposes, checked against a budget.

Farrell and Weir (2007) measured how much delay a myoelectric prosthesis user
tolerates between contracting the muscle and the device responding, and reported an
upper bound near 100 ms to 125 ms covering everything in the loop. Every stage in this
library reports the delay it imposes, but reporting is not enforcing: a caller can
assemble a band pass, an onset detector and a heavily smoothed amplitude estimator that
are each individually reasonable and whose sum is not. This module adds the sum and the
comparison.

Three properties make the accounting exact rather than indicative.

Delays add in cascade
    The group delay of a series connection is the sum of the group delays of its parts,
    because phase responses add and group delay is the negative derivative of phase. So
    the total is a plain sum and no interaction term is missing.

A filter has no single delay
    Group delay is a function of frequency. The fourth order band pass used here costs
    1.18 ms at 250 Hz and 31.68 ms at its 20 Hz corner, a factor of 27, and a quality
    factor 30 notch costs approaching 90 ms one Hertz from its centre. Charging the
    budget the value at one convenient frequency would be arbitrary, so
    :func:`filter_stage` takes a band and reduces the response over it to one number by
    a stated rule. Which rule is not a detail:

    ``power_weighted``, the default, is the group delay averaged over the band weighted
    by the squared magnitude response of the design, under the assumption that the
    input carries equal power at every frequency in the band. This is the delay the
    envelope of a broadband signal experiences, and the envelope is what a proportional
    controller acts on. It gives a notch almost no weight, correctly: at a transmission
    zero the filter is not delaying the signal, it is removing it.

    ``worst_case`` is the largest group delay anywhere in the pass band, where the pass
    band is the part of the requested band within ``passband_db`` of the design's
    maximum gain in it. This is a strict bound and it is a very conservative one when a
    notch is present, because the group delay of a notch is largest exactly where its
    magnitude is falling fastest, and a component there is losing most of its amplitude
    at the same time as it is being delayed. On the band pass and notch chain used in
    this project the two rules give 4.16 ms and 94.14 ms.

    The rule and the band are both recorded on the stage, so a total can never be read
    without the conditions under which it was computed.

Zero phase filtering is not free, it is unavailable
    A zero phase filter has exactly zero group delay because its reverse pass reads
    samples that have not been acquired. Entering it into a controller budget as a zero
    would produce a total that no controller can achieve, so :func:`filter_stage`
    refuses ``zero_phase`` rather than accepting it and charging nothing.

What the budget does not cover is whatever the caller does not enter into it. This
library implements conditioning, onset detection and amplitude estimation; the
classifier and the actuator are outside it, and their delays have to be supplied as
:func:`fixed_stage` entries. :func:`enforce` raises when the total exceeds the limit,
which is the check that was previously absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.envelope import EnvelopeEstimator
from myoelectric.algorithm.filters import FilterDesign, FilterMode
from myoelectric.algorithm.onset import OnsetDetector

__all__ = [
    "FARRELL_WEIR_LIMIT_MS",
    "DelayBudget",
    "DelayBudgetExceededError",
    "DelayStage",
    "FilterDelayRule",
    "assemble_budget",
    "detector_stage",
    "enforce",
    "envelope_stage",
    "filter_stage",
    "fixed_stage",
    "format_budget_table",
]

# Upper bound of the range Farrell and Weir (2007) report for the total delay from
# muscle contraction to device response. The lower end of their range is 100 ms; the
# upper end is used as the default limit so that a chain which fails this check fails
# under the most generous reading of the source.
FARRELL_WEIR_LIMIT_MS: float = 125.0

# How the frequency dependent group delay of a filter is reduced to the single number a
# budget can add up. See the module docstring for what each one means.
FilterDelayRule = Literal["power_weighted", "worst_case"]


class DelayBudgetExceededError(ValueError):
    """Raised when a chain's total delay exceeds the budget it was checked against."""


@dataclass(frozen=True, slots=True)
class DelayStage:
    """One contributor to the delay of a processing chain.

    Attributes:
        name: Identifier of the stage, as it appears in the budget table.
        delay_ms: Delay this stage imposes, in milliseconds. Never negative: a stage
            that advanced the signal in time would be non causal.
        basis: How the number was obtained, including any condition it depends on. A
            filter group delay depends on the band it was evaluated over, so the band
            belongs here rather than in a footnote.
    """

    name: str
    delay_ms: float
    basis: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.delay_ms):
            raise ValueError(f"stage {self.name!r} has a delay that is not finite")
        if self.delay_ms < 0.0:
            raise ValueError(
                f"stage {self.name!r} has a negative delay of {self.delay_ms} ms, "
                "which no causal stage can have"
            )


@dataclass(frozen=True, slots=True)
class DelayBudget:
    """The summed delay of a chain, compared against a limit."""

    stages: tuple[DelayStage, ...]
    limit_ms: float

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a budget needs at least one stage")
        if self.limit_ms <= 0.0:
            raise ValueError("limit_ms must be positive")

    @property
    def total_ms(self) -> float:
        """Total delay of the chain, the sum over its stages."""
        return float(sum(stage.delay_ms for stage in self.stages))

    @property
    def headroom_ms(self) -> float:
        """Delay still available inside the limit. Negative when the limit is exceeded."""
        return self.limit_ms - self.total_ms

    @property
    def within_budget(self) -> bool:
        """True when the total does not exceed the limit."""
        return self.total_ms <= self.limit_ms

    @property
    def dominant_stage(self) -> DelayStage:
        """The stage contributing the most delay, which is where to spend effort."""
        return max(self.stages, key=lambda stage: stage.delay_ms)


def fixed_stage(name: str, delay_ms: float, basis: str) -> DelayStage:
    """A stage whose delay is supplied rather than computed.

    Used for the parts of the loop this library does not implement, such as a
    classifier or the actuator, so that a budget can be completed rather than quietly
    left short.
    """
    return DelayStage(name=name, delay_ms=float(delay_ms), basis=basis)


def filter_stage(
    design: FilterDesign,
    band_hz: tuple[float, float],
    mode: FilterMode = "causal",
    rule: FilterDelayRule = "power_weighted",
    passband_db: float = 3.0,
    n_points: int = 2048,
) -> DelayStage:
    """Charge the budget the group delay of ``design`` over ``band_hz``.

    See the module docstring for what the two rules mean and why the default is the
    power weighted one. Frequencies at which the response is a transmission zero have
    no defined group delay and carry no signal, so they are excluded under either rule
    rather than making the whole stage undefined.

    Args:
        design: The filter whose delay is being charged.
        band_hz: The band over which the design has to carry signal.
        mode: Application mode. ``zero_phase`` is refused, see below.
        rule: ``power_weighted`` for the delay a broadband signal experiences,
            ``worst_case`` for a strict bound over the pass band.
        passband_db: Used by ``worst_case`` only. Frequencies attenuated by more than
            this relative to the design's maximum gain in the band are not treated as
            pass band. The default of 3 dB is the conventional corner definition.
        n_points: Size of the frequency grid the response is evaluated on.

    Raises:
        ValueError: If ``mode`` is ``zero_phase``, which is non causal and therefore
            cannot appear in a controller budget; if the band is empty or lies outside
            the sampled spectrum; if ``rule`` is not one of the two above; or if the
            response is a transmission zero everywhere in the band.
    """
    if mode == "zero_phase":
        raise ValueError(
            "a zero phase filter has zero group delay only because its reverse pass "
            "reads samples that have not been acquired, so it cannot be entered into a "
            "controller delay budget; apply the design causally instead"
        )
    low, high = float(band_hz[0]), float(band_hz[1])
    nyquist = 0.5 * design.sample_rate_hz
    if not 0.0 <= low < high <= nyquist:
        raise ValueError(
            f"band_hz must satisfy 0 <= low < high <= {nyquist} Hz, got {band_hz}"
        )
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    grid: NDArray[np.float64] = np.linspace(low, high, n_points, dtype=np.float64)
    delays = design.group_delay_ms(grid, mode=mode)
    magnitude = design.gain(grid, mode=mode)
    defined = np.isfinite(delays)
    if not bool(np.any(defined)):
        raise ValueError(
            f"{design.name} has no defined group delay anywhere in {low:g} to {high:g} Hz"
        )

    band_text = f"{low:g} to {high:g} Hz"
    if rule == "power_weighted":
        weight = np.where(defined, magnitude**2, 0.0)
        total = float(np.sum(weight))
        if total <= 0.0:
            raise ValueError(f"{design.name} passes no power in {band_text}")
        delay_ms = float(np.sum(weight * np.where(defined, delays, 0.0)) / total)
        basis = (
            f"group delay over {band_text}, weighted by the squared magnitude response "
            "under a flat input spectrum"
        )
    elif rule == "worst_case":
        peak_db = float(np.max(20.0 * np.log10(np.maximum(magnitude, 1e-15))))
        in_band = defined & (
            20.0 * np.log10(np.maximum(magnitude, 1e-15)) >= peak_db - float(passband_db)
        )
        if not bool(np.any(in_band)):
            raise ValueError(f"{design.name} has no pass band inside {band_text}")
        worst = int(np.argmax(np.where(in_band, delays, -np.inf)))
        delay_ms = float(delays[worst])
        basis = (
            f"largest group delay within {passband_db:g} dB of the peak gain over "
            f"{band_text}, reached at {grid[worst]:.2f} Hz"
        )
    else:
        raise ValueError(f"unknown rule: {rule!r}")

    return DelayStage(name=design.name, delay_ms=delay_ms, basis=basis)


def detector_stage(detector: OnsetDetector, sample_rate_hz: float) -> DelayStage:
    """Charge the budget an onset detector's decision delay.

    The decision delay, not the timing bias. The bias says where the detector places
    the onset and can be corrected for after the fact; the decision delay is how long
    the controller waits before it is told anything, and no correction recovers it.
    """
    return DelayStage(
        name=detector.name,
        delay_ms=1e3 * float(detector.decision_delay_s(sample_rate_hz)),
        basis=f"decision delay at {sample_rate_hz:g} Hz",
    )


def envelope_stage(estimator: EnvelopeEstimator, sample_rate_hz: float) -> DelayStage:
    """Charge the budget an amplitude estimator's design group delay at zero frequency.

    Zero frequency is the right point for this stage, unlike for a band pass, because
    what the estimator has to follow is a step change in contraction level and the step
    carries its energy at low frequency.
    """
    delay_samples = float(estimator.nominal_delay_samples(sample_rate_hz))
    return DelayStage(
        name=estimator.name,
        delay_ms=1e3 * delay_samples / float(sample_rate_hz),
        basis=f"design group delay at zero frequency, {delay_samples:.1f} samples",
    )


def assemble_budget(
    stages: tuple[DelayStage, ...], limit_ms: float = FARRELL_WEIR_LIMIT_MS
) -> DelayBudget:
    """Sum ``stages`` and compare the total against ``limit_ms``."""
    return DelayBudget(stages=stages, limit_ms=limit_ms)


def enforce(budget: DelayBudget) -> DelayBudget:
    """Return ``budget`` unchanged, or raise when it exceeds its limit.

    This is the function that turns a quoted budget into an enforced one. A caller that
    assembles a chain and passes it through here cannot ship a chain that is over
    budget without seeing the exception.

    Raises:
        DelayBudgetExceededError: When the total delay exceeds the limit.
    """
    if not budget.within_budget:
        worst = budget.dominant_stage
        raise DelayBudgetExceededError(
            f"chain delay {budget.total_ms:.1f} ms exceeds the {budget.limit_ms:.1f} ms "
            f"budget by {-budget.headroom_ms:.1f} ms; the largest single contributor is "
            f"{worst.name} at {worst.delay_ms:.1f} ms"
        )
    return budget


def format_budget_table(budget: DelayBudget) -> str:
    """Render a budget as a Markdown table with a total row and a verdict."""
    lines = [
        "| Stage | Delay (ms) | Basis |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| {stage.name} | {stage.delay_ms:.2f} | {stage.basis} |" for stage in budget.stages
    )
    verdict = "within budget" if budget.within_budget else "over budget"
    lines.append(f"| **total** | **{budget.total_ms:.2f}** | {verdict} |")
    lines.append(
        f"| budget | {budget.limit_ms:.2f} | Farrell and Weir (2007) upper bound, "
        f"headroom {budget.headroom_ms:+.2f} ms |"
    )
    return "\n".join(lines)
