"""Whether a record carries mains interference, and at which frequency.

Three things change when a real recording is substituted for the generator and
:mod:`myoelectric.pipeline.loaders` lists them. Two are declared by the dataset: the
sample rate is a property of the export and the meaning of an annotated onset is a
property of the protocol. The third is not. Mains interference is at 50 Hz in Europe and
60 Hz in North America, a dataset does not always say where it was recorded, and
:func:`~myoelectric.algorithm.filters.design_powerline_notch` designed at the wrong one
removes signal at three frequencies and leaves the interference where it was. This
module reads the answer off the record instead of assuming it.

What is measured is a ratio and not an amplitude. For each candidate, the power at its
fundamental and at each of its harmonics is compared against the power in the bins beside
that component, and each comparison is reported as decibels of excess over that local
floor. The floor is local because the surface electromyogram spectrum is not flat: it
peaks between 50 Hz and 150 Hz, which is exactly where the mains lines are, so a line
compared against the average power of the whole record would be judged against a floor it
does not sit on. Being a ratio also makes the verdict independent of the amplifier gain.

Four details make the ratio mean what it says.

The line is read from a window of bins
    A line that falls between two bins puts its power in both, and a mains supply drifts
    about its nominal frequency, so the line is the largest of the bin nearest the
    candidate and its two immediate neighbours rather than that bin alone.

The floor excludes every candidate, not only its own
    The bins beside 50 Hz include 60 Hz. Taking the floor there on a record that carries
    a 60 Hz line would raise the floor of the 50 Hz candidate and lower its excess, so
    every floor excludes the two bins either side of every candidate harmonic, which is
    the half width of the main lobe of the Hann window the spectrum is estimated with.
    What remains is reduced by its median rather than its mean, so the estimate stays a
    floor rather than an average over whatever else is in the neighbourhood.

The harmonics are reported, not aggregated
    The score of a candidate is the excess at its fundamental. Its harmonics are
    measured and reported beside it, because they say how many sections a notch has to
    have, but they are not summed into the score: mains interference falls away with
    each harmonic while the electromyogram spectrum it competes against rises towards
    its peak between 50 Hz and 150 Hz, so a harmonic buried in the signal is not
    evidence against a fundamental that is not.

The verdict can be that there is no answer
    A line is declared only when it stands ``min_excess_db`` above both its own local
    floor and every other candidate. A record with no interference and a record carrying
    two lines of equal strength are both reported as such, rather than resolved by the
    order in which the candidates were listed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from myoelectric.algorithm.features_freq import PowerSpectrum, welch_spectrum

__all__ = [
    "PowerlineCandidate",
    "PowerlineCheck",
    "check_powerline",
    "format_powerline_table",
]

# Bins the line itself is read from: the nearest bin and its two neighbours, which
# covers both the offset between the line and the bin grid and the drift of a supply.
_LINE_HALFWIDTH_BINS: int = 1

# A Hann window has a main lobe four bins wide, so a line raises the two bins either
# side of its own. Those belong to the line and never to a floor.
_LOBE_HALFWIDTH_BINS: int = 2

# Bins outside the main lobe and within this distance form the floor a line is measured
# against. Ten bins is wide enough for a median to be a stable estimate and narrow
# enough for it to stay local: a surface electromyogram spectrum changes little over
# 10 Hz and a great deal over 100 Hz.
_FLOOR_HALFWIDTH_BINS: int = 10


@dataclass(frozen=True, slots=True)
class PowerlineCandidate:
    """One candidate mains frequency, measured against the floor beside it.

    Attributes:
        fundamental_hz: The candidate.
        harmonics_hz: Harmonics that were measured, the fundamental first. Components at
            or above the Nyquist frequency are not in the record and are dropped.
        harmonic_excess_db: ``10 log10(line / floor)`` at each of those harmonics, in
            the same order.
    """

    fundamental_hz: float
    harmonics_hz: tuple[float, ...]
    harmonic_excess_db: tuple[float, ...]

    @property
    def excess_db(self) -> float:
        """Excess at the fundamental, which is what the verdict is decided on."""
        return self.harmonic_excess_db[0]


@dataclass(frozen=True, slots=True)
class PowerlineCheck:
    """The verdict on one record, with the measurement behind it.

    Attributes:
        candidates: One entry per candidate, in the order they were supplied.
        detected_hz: The mains frequency, or ``None`` when no candidate stood far enough
            above both the floor and the others to be named.
        reason: The verdict in words, including the numbers it was reached on.
        min_excess_db: Margin the verdict required.
        resolution_hz: Frequency resolution the spectrum was estimated at, which sets
            how wide the line window and the floor are in hertz.
    """

    candidates: tuple[PowerlineCandidate, ...]
    detected_hz: float | None
    reason: str
    min_excess_db: float
    resolution_hz: float

    @property
    def best(self) -> PowerlineCandidate:
        """Candidate with the largest excess, named or not."""
        return max(self.candidates, key=lambda candidate: candidate.excess_db)


def _harmonics_below_nyquist(
    fundamental_hz: float, n_harmonics: int, nyquist_hz: float
) -> tuple[float, ...]:
    """Harmonics of ``fundamental_hz`` that the record can carry."""
    frequencies = (fundamental_hz * (index + 1) for index in range(n_harmonics))
    return tuple(frequency for frequency in frequencies if frequency < nyquist_hz)


def _nearest_bin(spectrum: PowerSpectrum, frequency_hz: float) -> int:
    """Index of the bin closest to ``frequency_hz``."""
    return int(np.argmin(np.abs(spectrum.frequencies_hz - frequency_hz)))


def _window_mask(size: int, centre: int, halfwidth: int) -> NDArray[np.bool_]:
    """Bins within ``halfwidth`` of ``centre``, clipped to the grid."""
    mask = np.zeros(size, dtype=np.bool_)
    mask[max(0, centre - halfwidth) : centre + halfwidth + 1] = True
    return mask


def check_powerline(
    x: NDArray[np.float64],
    sample_rate_hz: float,
    candidates: tuple[float, ...] = (50.0, 60.0),
    n_harmonics: int = 3,
    min_excess_db: float = 6.0,
    resolution_hz: float = 1.0,
) -> PowerlineCheck:
    """Decide which mains frequency, if any, a record carries.

    Args:
        x: One channel, as delivered by
            :meth:`~myoelectric.pipeline.loaders.EmgRecording.channel`. Conditioning
            filters must not have been applied yet: a notch removes the very line this
            is looking for.
        sample_rate_hz: Sample rate of the record.
        candidates: Mains frequencies to consider. The two in use worldwide are the
            default; a single candidate turns the check into a yes or no question about
            that frequency.
        n_harmonics: Number of components per candidate, counted from the fundamental.
            They are reported rather than scored, and they are excluded from every
            floor, so raising this does not make a line easier to declare.
        min_excess_db: Margin required over both the local floor and the next best
            candidate. The default of 6 dB is a factor of four in power, which is above
            the bin to bin variation of a Welch estimate on a record of a few seconds
            and below the excess that interference worth notching produces. How far
            above depends on how many segments the record affords, so a short record
            wants a larger margin.
        resolution_hz: Frequency resolution the spectrum is estimated at, which fixes
            the segment length at ``sample_rate_hz / resolution_hz`` samples. The
            default of 1 Hz reads a line from one bin either side of the candidate and
            its floor from ten, both far inside the 10 Hz that separates the two
            candidates, and needs a record of at least one second.

    Raises:
        ValueError: If a setting is out of range; if a candidate is at or above the
            Nyquist frequency, where the record cannot carry it; if the record is too
            short for the requested resolution, which Welch would otherwise meet by
            coarsening the grid; if the candidates are packed so closely that a harmonic
            has no neighbouring bins left to form a floor from; or if the record carries
            no power at or beside a candidate at all.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not candidates:
        raise ValueError("at least one candidate mains frequency is needed")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    if resolution_hz <= 0.0:
        raise ValueError("resolution_hz must be positive")
    nyquist_hz = 0.5 * sample_rate_hz
    for fundamental_hz in candidates:
        if not 0.0 < fundamental_hz < nyquist_hz:
            raise ValueError(
                f"candidate {fundamental_hz:g} Hz must lie strictly between 0 and "
                f"{nyquist_hz:g} Hz, which is the range the record can carry"
            )

    # welch_spectrum shortens its segment to fit a short record, so without this the
    # answer would come back on a coarser grid than the one that was asked for, and on a
    # grid coarse enough the two candidates share a line window.
    segment_samples = round(sample_rate_hz / resolution_hz)
    if samples.size < segment_samples:
        raise ValueError(
            f"a record of {samples.size} samples resolves "
            f"{sample_rate_hz / max(samples.size, 1):.3g} Hz, which is coarser than the "
            f"requested {resolution_hz:g} Hz; at least {segment_samples} samples are "
            "needed to separate a mains line from the floor beside it"
        )

    spectrum = welch_spectrum(samples, sample_rate_hz, segment_s=1.0 / resolution_hz)
    harmonics = {
        fundamental_hz: _harmonics_below_nyquist(fundamental_hz, n_harmonics, nyquist_hz)
        for fundamental_hz in candidates
    }
    n_bins = int(spectrum.power.size)
    in_a_lobe = np.zeros(n_bins, dtype=np.bool_)
    for frequencies in harmonics.values():
        for frequency_hz in frequencies:
            in_a_lobe |= _window_mask(
                n_bins, _nearest_bin(spectrum, frequency_hz), _LOBE_HALFWIDTH_BINS
            )

    measured: list[PowerlineCandidate] = []
    for fundamental_hz in candidates:
        excesses: list[float] = []
        for frequency_hz in harmonics[fundamental_hz]:
            centre = _nearest_bin(spectrum, frequency_hz)
            line = _window_mask(n_bins, centre, _LINE_HALFWIDTH_BINS)
            beside = _window_mask(n_bins, centre, _FLOOR_HALFWIDTH_BINS) & ~in_a_lobe
            if not bool(np.any(beside)):
                raise ValueError(
                    f"there are no bins beside {frequency_hz:g} Hz that do not belong to "
                    "a candidate, so its floor cannot be measured; the candidates are "
                    "packed more closely than the resolution can separate"
                )
            line_power = float(np.max(spectrum.power[line]))
            floor_power = float(np.median(spectrum.power[beside]))
            if line_power <= 0.0 or floor_power <= 0.0:
                raise ValueError(
                    f"the record carries no power at or beside {frequency_hz:g} Hz, so a "
                    "line cannot be measured against a floor there"
                )
            excesses.append(10.0 * math.log10(line_power / floor_power))
        measured.append(
            PowerlineCandidate(
                fundamental_hz=fundamental_hz,
                harmonics_hz=harmonics[fundamental_hz],
                harmonic_excess_db=tuple(excesses),
            )
        )

    ranked = sorted(measured, key=lambda candidate: candidate.excess_db, reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    detected_hz: float | None = None
    if best.excess_db < min_excess_db:
        reason = (
            f"No candidate stands {min_excess_db:g} dB above the local floor; the "
            f"largest excess is {best.excess_db:.1f} dB at {best.fundamental_hz:g} Hz"
        )
    elif runner_up is not None and best.excess_db - runner_up.excess_db < min_excess_db:
        reason = (
            f"{best.fundamental_hz:g} Hz at {best.excess_db:.1f} dB and "
            f"{runner_up.fundamental_hz:g} Hz at {runner_up.excess_db:.1f} dB are within "
            f"{min_excess_db:g} dB of each other, so the mains frequency is ambiguous"
        )
    else:
        detected_hz = best.fundamental_hz
        reason = f"{best.fundamental_hz:g} Hz stands {best.excess_db:.1f} dB above the local floor"
        if runner_up is not None:
            reason += (
                f" and {best.excess_db - runner_up.excess_db:.1f} dB above "
                f"{runner_up.fundamental_hz:g} Hz"
            )

    return PowerlineCheck(
        candidates=tuple(measured),
        detected_hz=detected_hz,
        reason=reason,
        min_excess_db=min_excess_db,
        resolution_hz=spectrum.resolution_hz,
    )


def format_powerline_table(check: PowerlineCheck) -> str:
    """Render a check as a Markdown table followed by its verdict.

    The excess column carries one figure per harmonic, the fundamental first, because
    the profile across the harmonics is what says how many notch sections a record
    needs.
    """
    lines = [
        "| Candidate (Hz) | Harmonics | Excess over the local floor (dB) |",
        "| ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {candidate.fundamental_hz:g} | {len(candidate.harmonics_hz)} | "
        f"{', '.join(f'{excess:.1f}' for excess in candidate.harmonic_excess_db)} |"
        for candidate in check.candidates
    )
    lines.append("")
    lines.append(f"{check.reason}.")
    return "\n".join(lines)
