"""Filter design and application for surface electromyography.

Three designs cover the contaminations that a surface recording carries.

Band pass, 20 Hz to 450 Hz, fourth order Butterworth
    Surface electromyogram power is concentrated between about 20 Hz and 500 Hz, with
    the peak of the spectrum between 50 Hz and 150 Hz. De Luca et al. (2010) measured
    the trade off between movement artefact rejection and signal loss directly and
    recommend a high pass corner at 20 Hz for surface recordings, which removes the
    bulk of the artefact while discarding a small fraction of the signal power. The
    upper corner is placed at 450 Hz so that it sits below the Nyquist frequency of a
    1000 Hz record and well below that of the 2000 Hz records used here, while
    retaining the fast rising edges of individual motor unit potentials. Butterworth
    is chosen because its pass band is maximally flat, so the amplitude features
    computed after filtering are not distorted by pass band ripple.

Power line notch, 50 Hz fundamental with harmonics, cascaded second order sections
    Mains interference is narrowband and sits inside the electromyogram pass band, so
    it cannot be removed by adjusting the band pass corners. Each component is removed
    by one second order notch section with a stated quality factor. A quality factor
    of 30 gives a minus 3 dB width of 50 / 30, that is 1.7 Hz, at the fundamental,
    which is narrow enough to leave the surrounding signal essentially untouched.
    Harmonics above the Nyquist frequency are discarded rather than aliased.

High pass, 20 Hz, fourth order Butterworth
    Provided separately for pipelines that need movement artefact removal without a
    band limit, for example when the full high frequency content is required for a
    later spectral analysis. Same corner and same rationale as the lower corner of the
    band pass.

Two application modes are provided and the difference between them is not cosmetic.

``causal``
    A single forward pass through the second order sections. The output at sample
    ``n`` depends only on inputs up to sample ``n``, so the filter can run on a
    prosthesis controller. It imposes the group delay of the design, which for a
    fourth order Butterworth band pass at 20 Hz to 450 Hz sampled at 2000 Hz is a few
    milliseconds in the middle of the pass band and considerably more near the
    corners.

``zero_phase``
    A forward pass followed by a reverse pass over the already filtered signal, using
    the initial state construction of Gustafsson (1996). The two passes have equal and
    opposite phase responses, so the combined phase response is exactly zero and no
    feature is displaced in time. The squared magnitude response also means the
    effective attenuation is doubled in decibels and the effective order is doubled.

    Zero phase filtering is legitimate whenever the whole record is already available
    and the question being asked is about the record as a whole: offline spectral
    analysis, fatigue tracking across a stored contraction, or the production of
    reference onset annotations against which a causal detector is later judged.

    Zero phase filtering is not legitimate in a real time prosthesis controller. The
    reverse pass reads samples that have not yet been acquired, so the algorithm is
    non causal and cannot be implemented in a stream. Reporting a controller latency
    measured under zero phase filtering understates the true latency, and reporting an
    onset detection bias measured under zero phase filtering removes a delay that the
    real controller would still incur. The functions below therefore keep the two
    modes separate and :func:`group_delay_samples` returns exactly zero only for the
    zero phase mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

__all__ = [
    "FilterDesign",
    "FilterMode",
    "apply_causal",
    "apply_filter",
    "apply_zero_phase",
    "cascade",
    "design_bandpass",
    "design_highpass",
    "design_lowpass",
    "design_powerline_notch",
    "group_delay_samples",
    "phase_delay_samples",
]

FilterMode = Literal["causal", "zero_phase"]

# Magnitude below which a response is treated as a transmission zero, so that phase
# derived quantities are reported as undefined rather than as a large arbitrary number.
# Minus 120 dB is far below the dynamic range of any electromyography amplifier.
_ZERO_MAGNITUDE: float = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class FilterDesign:
    """A digital filter stored as cascaded second order sections.

    Second order sections are used throughout rather than transfer function
    coefficients because a direct form transfer function of order eight or above loses
    accuracy when its roots are clustered near the unit circle, which is exactly the
    case for a narrow notch.
    """

    name: str
    sos: NDArray[np.float64]
    sample_rate_hz: float
    rationale: str

    def __post_init__(self) -> None:
        if self.sos.ndim != 2 or self.sos.shape[1] != 6:
            raise ValueError(f"sos must have shape (n_sections, 6), got {self.sos.shape}")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def n_sections(self) -> int:
        """Number of cascaded second order sections."""
        return int(self.sos.shape[0])

    @property
    def order(self) -> int:
        """Order of the cascade counted as two per second order section."""
        return 2 * self.n_sections

    def gain(
        self, frequencies_hz: NDArray[np.float64], mode: FilterMode = "causal"
    ) -> NDArray[np.float64]:
        """Magnitude response at ``frequencies_hz``.

        In ``zero_phase`` mode the magnitude is squared, because the signal passes
        through the same design twice.
        """
        _, response = sp_signal.sosfreqz(
            self.sos, worN=frequencies_hz, fs=self.sample_rate_hz
        )
        magnitude = np.abs(np.asarray(response, dtype=np.complex128))
        if mode == "zero_phase":
            magnitude = magnitude**2
        return np.asarray(magnitude, dtype=np.float64)

    def gain_db(
        self, frequencies_hz: NDArray[np.float64], mode: FilterMode = "causal"
    ) -> NDArray[np.float64]:
        """Magnitude response in decibels, floored at minus 300 dB."""
        magnitude = np.maximum(self.gain(frequencies_hz, mode=mode), 1e-15)
        return np.asarray(20.0 * np.log10(magnitude), dtype=np.float64)

    def group_delay_samples(
        self, frequencies_hz: NDArray[np.float64], mode: FilterMode = "causal"
    ) -> NDArray[np.float64]:
        """Group delay in samples at ``frequencies_hz``, zero for ``zero_phase``."""
        return group_delay_samples(self, frequencies_hz, mode=mode)

    def group_delay_ms(
        self, frequencies_hz: NDArray[np.float64], mode: FilterMode = "causal"
    ) -> NDArray[np.float64]:
        """Group delay in milliseconds at ``frequencies_hz``."""
        delay = self.group_delay_samples(frequencies_hz, mode=mode)
        return np.asarray(1e3 * delay / self.sample_rate_hz, dtype=np.float64)


def _validate_corner(name: str, value: float, sample_rate_hz: float) -> float:
    nyquist = 0.5 * sample_rate_hz
    if not 0.0 < value < nyquist:
        raise ValueError(f"{name} must lie strictly between 0 and {nyquist} Hz, got {value}")
    return value


def design_bandpass(
    sample_rate_hz: float,
    low_hz: float = 20.0,
    high_hz: float = 450.0,
    order: int = 4,
) -> FilterDesign:
    """Butterworth band pass covering the surface electromyogram band.

    Args:
        sample_rate_hz: Sample rate of the record.
        low_hz: Lower corner. The default of 20 Hz follows De Luca et al. (2010).
        high_hz: Upper corner, clipped to just below the Nyquist frequency.
        order: Butterworth order of the band pass, counted as in ``scipy.signal.butter``
            so the realised cascade has ``2 * order`` poles.
    """
    nyquist = 0.5 * sample_rate_hz
    high = min(high_hz, 0.98 * nyquist)
    _validate_corner("low_hz", low_hz, sample_rate_hz)
    _validate_corner("high_hz", high, sample_rate_hz)
    if low_hz >= high:
        raise ValueError("low_hz must be below high_hz")
    sos = sp_signal.butter(order, [low_hz, high], btype="bandpass", output="sos", fs=sample_rate_hz)
    return FilterDesign(
        name=f"bandpass {low_hz:g}-{high:g} Hz order {order} Butterworth",
        sos=np.asarray(sos, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        rationale=(
            "Retains the 20 Hz to 450 Hz band that carries surface electromyogram power. "
            "The lower corner rejects movement artefact, the upper corner rejects "
            "out of band instrumentation noise, and the Butterworth response keeps the "
            "pass band flat so amplitude features are not distorted."
        ),
    )


def design_highpass(
    sample_rate_hz: float, cutoff_hz: float = 20.0, order: int = 4
) -> FilterDesign:
    """Butterworth high pass for movement artefact removal."""
    _validate_corner("cutoff_hz", cutoff_hz, sample_rate_hz)
    sos = sp_signal.butter(order, cutoff_hz, btype="highpass", output="sos", fs=sample_rate_hz)
    return FilterDesign(
        name=f"highpass {cutoff_hz:g} Hz order {order} Butterworth",
        sos=np.asarray(sos, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        rationale=(
            "Removes electrode and cable movement artefact, whose energy lies below "
            "about 20 Hz, without imposing an upper band limit."
        ),
    )


def design_lowpass(
    sample_rate_hz: float, cutoff_hz: float, order: int = 2
) -> FilterDesign:
    """Butterworth low pass, used to smooth a rectified signal into an envelope."""
    _validate_corner("cutoff_hz", cutoff_hz, sample_rate_hz)
    sos = sp_signal.butter(order, cutoff_hz, btype="lowpass", output="sos", fs=sample_rate_hz)
    return FilterDesign(
        name=f"lowpass {cutoff_hz:g} Hz order {order} Butterworth",
        sos=np.asarray(sos, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        rationale=(
            "Smooths a rectified signal into an amplitude estimate. The corner sets the "
            "trade off between residual ripple and the delay imposed on the estimate."
        ),
    )


def design_powerline_notch(
    sample_rate_hz: float,
    fundamental_hz: float = 50.0,
    n_harmonics: int = 3,
    quality: float = 30.0,
) -> FilterDesign:
    """Cascade of notches at the mains fundamental and its harmonics.

    Args:
        sample_rate_hz: Sample rate of the record.
        fundamental_hz: Mains frequency.
        n_harmonics: Number of components including the fundamental. Components at or
            above the Nyquist frequency are silently discarded, because a notch cannot
            be placed there.
        quality: Quality factor of each section. The minus 3 dB width of a section at
            centre frequency ``f`` is ``f / quality``.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    if quality <= 0.0:
        raise ValueError("quality must be positive")
    nyquist = 0.5 * sample_rate_hz
    sections: list[NDArray[np.float64]] = []
    centres: list[float] = []
    for harmonic in range(1, n_harmonics + 1):
        centre = fundamental_hz * harmonic
        if centre >= 0.98 * nyquist:
            break
        numerator, denominator = sp_signal.iirnotch(centre, quality, fs=sample_rate_hz)
        sections.append(np.asarray(sp_signal.tf2sos(numerator, denominator), dtype=np.float64))
        centres.append(centre)
    if not sections:
        raise ValueError("no notch component lies below the Nyquist frequency")
    sos = np.concatenate(sections, axis=0)
    listed = ", ".join(f"{c:g}" for c in centres)
    return FilterDesign(
        name=f"powerline notch at {listed} Hz, quality {quality:g}",
        sos=sos,
        sample_rate_hz=sample_rate_hz,
        rationale=(
            "Mains interference is narrowband and lies inside the electromyogram pass "
            "band, so it must be removed by notches rather than by moving the band pass "
            f"corners. Each section is {fundamental_hz / quality:.2f} Hz wide at the "
            "fundamental, which leaves neighbouring signal components intact."
        ),
    )


def cascade(designs: tuple[FilterDesign, ...], name: str, rationale: str) -> FilterDesign:
    """Concatenate the sections of several designs into one design.

    Group delays of cascaded filters add, and magnitude responses multiply, so the
    cascade is exactly the series connection of its parts.
    """
    if not designs:
        raise ValueError("cascade requires at least one design")
    rates = {design.sample_rate_hz for design in designs}
    if len(rates) != 1:
        raise ValueError("cannot cascade designs with different sample rates")
    sos = np.concatenate([design.sos for design in designs], axis=0)
    return FilterDesign(
        name=name,
        sos=np.asarray(sos, dtype=np.float64),
        sample_rate_hz=designs[0].sample_rate_hz,
        rationale=rationale,
    )


def apply_causal(design: FilterDesign, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Filter ``x`` forward only.

    The output at sample ``n`` depends on inputs up to sample ``n`` and on nothing
    later, so this mode is the one that a real time controller can run.
    """
    return np.asarray(sp_signal.sosfilt(design.sos, x), dtype=np.float64)


def apply_zero_phase(design: FilterDesign, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Filter ``x`` forward and then backward, giving exactly zero phase response.

    Non causal. See the module docstring for when this is and is not legitimate.
    """
    return np.asarray(sp_signal.sosfiltfilt(design.sos, x), dtype=np.float64)


def apply_filter(
    design: FilterDesign, x: NDArray[np.float64], mode: FilterMode = "causal"
) -> NDArray[np.float64]:
    """Apply ``design`` to ``x`` in the requested mode."""
    if mode == "causal":
        return apply_causal(design, x)
    if mode == "zero_phase":
        return apply_zero_phase(design, x)
    raise ValueError(f"unknown filter mode: {mode!r}")


def group_delay_samples(
    design: FilterDesign,
    frequencies_hz: NDArray[np.float64],
    mode: FilterMode = "causal",
) -> NDArray[np.float64]:
    """Group delay of ``design`` in samples.

    The group delay of a cascade is the sum of the group delays of its sections, so
    each second order section is converted to its own transfer function pair and
    evaluated separately. This avoids expanding a high order cascade into a single
    transfer function, which loses accuracy when poles are clustered.

    In ``zero_phase`` mode the result is exactly zero at every frequency, because the
    forward and reverse passes have equal and opposite phase responses.

    Group delay is undefined at a transmission zero, because the phase of a response of
    zero magnitude carries no information. A notch places such a zero exactly on the
    unit circle at its centre frequency, so the result is ``nan`` at any frequency where
    the cascade magnitude falls below ``_ZERO_MAGNITUDE``, rather than the arbitrarily
    large number that differentiating a vanishing response returns.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if mode == "zero_phase":
        return np.zeros_like(frequencies)
    if mode != "causal":
        raise ValueError(f"unknown filter mode: {mode!r}")
    defined = design.gain(frequencies, mode="causal") > _ZERO_MAGNITUDE
    total = np.full(frequencies.shape, np.nan, dtype=np.float64)
    if not bool(np.any(defined)):
        return total
    usable = frequencies[defined]
    partial = np.zeros_like(usable)
    for section in design.sos:
        numerator = np.asarray(section[:3], dtype=np.float64)
        denominator = np.asarray(section[3:], dtype=np.float64)
        _, delay = sp_signal.group_delay(
            (numerator, denominator), w=usable, fs=design.sample_rate_hz
        )
        partial += np.asarray(delay, dtype=np.float64)
    total[defined] = partial
    return total


def phase_delay_samples(
    design: FilterDesign,
    frequencies_hz: NDArray[np.float64],
    mode: FilterMode = "causal",
) -> NDArray[np.float64]:
    """Phase delay in samples, which is the shift a steady sinusoid experiences.

    Group delay describes the shift of an envelope; phase delay describes the shift of
    a carrier. Both are reported because a test that measures the displacement of a
    pure sinusoid measures the phase delay, while a test that measures the
    displacement of a burst measures the group delay.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if mode == "zero_phase":
        return np.zeros_like(frequencies)
    if mode != "causal":
        raise ValueError(f"unknown filter mode: {mode!r}")
    _, response = sp_signal.sosfreqz(design.sos, worN=frequencies, fs=design.sample_rate_hz)
    phase = np.unwrap(np.angle(np.asarray(response, dtype=np.complex128)))
    omega = 2.0 * np.pi * frequencies / design.sample_rate_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(omega > 0.0, -phase / omega, 0.0)
    return np.asarray(delay, dtype=np.float64)
