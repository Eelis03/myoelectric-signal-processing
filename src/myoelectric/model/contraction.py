"""Contraction and gesture definitions expressed as a neural excitation profile."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

__all__ = ["ContractionEvent", "ContractionProfile"]


@dataclass(frozen=True, slots=True)
class ContractionEvent:
    """One trapezoidal contraction.

    Excitation rises linearly from the resting level to ``plateau_excitation`` over
    ``rise_s`` starting at ``onset_s``, holds, then falls linearly to rest over
    ``fall_s``, reaching rest at ``offset_s``.

    Attributes:
        onset_s: Time at which excitation starts to rise. This is the neural onset.
        offset_s: Time at which excitation returns to rest.
        plateau_excitation: Normalised excitation at the plateau, in ``(0, 1]``. The
            value is compared directly with motor unit recruitment thresholds, which
            are also normalised to ``(0, 1]``.
        rise_s: Duration of the rising ramp.
        fall_s: Duration of the falling ramp.
        label: Name of the gesture or contraction, carried through for reporting.
    """

    onset_s: float
    offset_s: float
    plateau_excitation: float
    rise_s: float = 0.1
    fall_s: float = 0.1
    label: str = "contraction"

    def __post_init__(self) -> None:
        if self.offset_s <= self.onset_s:
            raise ValueError("offset_s must follow onset_s")
        if not 0.0 < self.plateau_excitation <= 1.0:
            raise ValueError(
                f"plateau_excitation must lie in (0, 1], got {self.plateau_excitation}"
            )
        if self.rise_s < 0.0 or self.fall_s < 0.0:
            raise ValueError("rise_s and fall_s must not be negative")
        if self.rise_s + self.fall_s > self.offset_s - self.onset_s:
            raise ValueError("the ramps do not fit inside the event")

    @property
    def duration_s(self) -> float:
        """Total duration of the event including both ramps."""
        return self.offset_s - self.onset_s


@dataclass(frozen=True, slots=True)
class ContractionProfile:
    """A sequence of non overlapping contraction events on a resting background."""

    events: tuple[ContractionEvent, ...]
    rest_excitation: float = 0.0

    def __post_init__(self) -> None:
        if self.rest_excitation < 0.0:
            raise ValueError("rest_excitation must not be negative")
        ordered = sorted(self.events, key=lambda event: event.onset_s)
        for earlier, later in pairwise(ordered):
            if later.onset_s < earlier.offset_s:
                raise ValueError("contraction events must not overlap")

    @classmethod
    def single(
        cls,
        onset_s: float,
        offset_s: float,
        plateau_excitation: float = 0.5,
        rise_s: float = 0.1,
        fall_s: float = 0.1,
        label: str = "contraction",
    ) -> ContractionProfile:
        """Convenience constructor for a profile holding exactly one event."""
        return cls(
            events=(
                ContractionEvent(
                    onset_s=onset_s,
                    offset_s=offset_s,
                    plateau_excitation=plateau_excitation,
                    rise_s=rise_s,
                    fall_s=fall_s,
                    label=label,
                ),
            )
        )

    @classmethod
    def rest_only(cls) -> ContractionProfile:
        """A profile with no contraction, used to measure false positive rates."""
        return cls(events=())

    def excitation(self, times_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalised excitation sampled at ``times_s``."""
        values = np.full(times_s.shape, self.rest_excitation, dtype=np.float64)
        for event in self.events:
            rise_end = event.onset_s + event.rise_s
            fall_start = event.offset_s - event.fall_s
            ramp_up = (
                np.clip((times_s - event.onset_s) / event.rise_s, 0.0, 1.0)
                if event.rise_s > 0.0
                else (times_s >= event.onset_s).astype(np.float64)
            )
            ramp_down = (
                np.clip((event.offset_s - times_s) / event.fall_s, 0.0, 1.0)
                if event.fall_s > 0.0
                else (times_s < event.offset_s).astype(np.float64)
            )
            inside = (times_s >= event.onset_s) & (times_s < event.offset_s)
            shape = np.where(times_s < rise_end, ramp_up, 1.0)
            shape = np.where(times_s >= fall_start, ramp_down, shape)
            values = np.where(
                inside,
                np.maximum(values, self.rest_excitation + shape * event.plateau_excitation),
                values,
            )
        return values

    @property
    def onset_times_s(self) -> tuple[float, ...]:
        """Neural onset time of every event, in order."""
        return tuple(event.onset_s for event in sorted(self.events, key=lambda e: e.onset_s))

    @property
    def offset_times_s(self) -> tuple[float, ...]:
        """Offset time of every event, in order."""
        return tuple(event.offset_s for event in sorted(self.events, key=lambda e: e.onset_s))
