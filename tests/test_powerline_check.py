"""Reading the mains frequency of a record off the record itself.

The mains frequency is one of the three things :mod:`myoelectric.pipeline.loaders` says
must be checked when a real recording is substituted, and it is the one a dataset does
not always declare. What is tested here is that the answer is reached on evidence: that
the line is found at the frequency it was injected at, that a record carrying none is
reported as carrying none, that two lines of equal strength are reported as ambiguous
rather than resolved by the order of the candidate list, and that the excess moves with
the amplitude of the line by the factor its definition requires.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from myoelectric.algorithm.filters import apply_causal, design_powerline_notch
from myoelectric.analysis.powerline_check import (
    check_powerline,
    format_powerline_table,
)
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate
from tests.helpers import sine


def _record(
    lines: tuple[tuple[float, float], ...],
    sample_rate_hz: float,
    duration_s: float,
    seed: int,
) -> NDArray[np.float64]:
    """White noise of unit variance plus one sine per ``(frequency, amplitude)`` pair.

    White noise gives a flat floor whose level is known, so the excess the check reports
    can be predicted rather than only compared between records.
    """
    sampling = SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=duration_s)
    samples = np.asarray(np.random.default_rng(seed).standard_normal(sampling.n_samples))
    for frequency_hz, amplitude in lines:
        samples = samples + sine(frequency_hz, sample_rate_hz, duration_s, amplitude)
    return np.asarray(samples, dtype=np.float64)


@pytest.mark.parametrize("fundamental_hz", [50.0, 60.0])
def test_a_line_is_found_at_the_frequency_it_was_injected_at(
    fundamental_hz: float, sample_rate_hz: float
) -> None:
    """Naming the wrong one costs signal and leaves the interference in place."""
    check = check_powerline(
        _record(((fundamental_hz, 0.5),), sample_rate_hz, 4.0, seed=11), sample_rate_hz
    )
    assert check.detected_hz == fundamental_hz
    assert check.best.fundamental_hz == fundamental_hz
    assert f"{fundamental_hz:g} Hz stands" in check.reason


def test_a_record_without_a_line_reports_that_there_is_none(sample_rate_hz: float) -> None:
    """A check that always names a frequency has not checked anything."""
    check = check_powerline(_record((), sample_rate_hz, 4.0, seed=13), sample_rate_hz)
    assert check.detected_hz is None
    assert all(candidate.excess_db < check.min_excess_db for candidate in check.candidates)
    assert "above the local floor" in check.reason


def test_no_line_is_declared_on_sixty_records_that_carry_none(sample_rate_hz: float) -> None:
    """The threshold has to sit above the fluctuation of the estimate itself.

    A Welch estimate of a record with no line still has a largest bin, and the bin at a
    candidate is above the median beside it about half the time, so the question is how
    far above 6 dB is. Sixty independent noise records answer it as a rate: none is
    declared, which by the rule of three has an upper 95 per cent bound of 0.050.
    """
    declared = [
        check_powerline(_record((), sample_rate_hz, 4.0, seed=seed), sample_rate_hz).detected_hz
        for seed in range(60)
    ]
    assert [frequency for frequency in declared if frequency is not None] == []


def test_two_lines_of_equal_strength_are_reported_as_ambiguous(sample_rate_hz: float) -> None:
    """Both stand well above the floor, so the floor alone cannot separate them.

    Without the margin condition the answer would be whichever candidate the caller
    happened to list first, which is not a measurement.
    """
    record = _record(((50.0, 0.5), (60.0, 0.5)), sample_rate_hz, 4.0, seed=17)
    check = check_powerline(record, sample_rate_hz)
    assert check.detected_hz is None
    assert "ambiguous" in check.reason
    assert min(candidate.excess_db for candidate in check.candidates) > check.min_excess_db


def test_the_excess_follows_the_amplitude_of_the_line(sample_rate_hz: float) -> None:
    """Power is amplitude squared, so twice the amplitude is 20 log10(2) = 6.02 dB more.

    Both records are built from the same seed, so the floor is the same realisation in
    each and the only difference between them is the line. The tolerance is the floor's
    own contribution to the line bin. The line and the floor add there as complex
    amplitudes, so the measured power lies between ``(a - b)^2`` and ``(a + b)^2`` for
    line amplitude ``a`` and floor amplitude ``b`` by the Cauchy Schwarz inequality, and
    the ratio of the two records is bounded by ``20 log10((1 + b/a) / (1 - b/a))``. The
    ratio ``b / a`` is read from the excess of the quieter record, which is
    ``20 log10(1 + a / b)``, so the tolerance is derived from the measurement rather
    than from the error observed.
    """
    quiet = check_powerline(
        _record(((50.0, 1.0),), sample_rate_hz, 4.0, seed=19), sample_rate_hz, n_harmonics=1
    )
    loud = check_powerline(
        _record(((50.0, 2.0),), sample_rate_hz, 4.0, seed=19), sample_rate_hz, n_harmonics=1
    )
    ratio = 10.0 ** (quiet.best.excess_db / 20.0) - 1.0
    tolerance = 20.0 * math.log10((1.0 + 1.0 / ratio) / (1.0 - 1.0 / ratio))
    assert loud.best.excess_db - quiet.best.excess_db == pytest.approx(
        20.0 * math.log10(2.0), abs=tolerance
    )


@pytest.mark.parametrize("gain", [0.25, 8.0])
def test_the_excess_is_a_ratio_and_does_not_move_with_the_gain(
    gain: float, sample_rate_hz: float
) -> None:
    """An amplifier setting must not change the verdict, so the statistic is a ratio."""
    record = _record(((50.0, 0.5),), sample_rate_hz, 4.0, seed=23)
    plain = check_powerline(record, sample_rate_hz)
    amplified = check_powerline(gain * record, sample_rate_hz)
    assert amplified.detected_hz == plain.detected_hz
    assert amplified.best.excess_db == pytest.approx(plain.best.excess_db, rel=1e-12)


def test_harmonics_above_the_nyquist_frequency_are_dropped(sample_rate_hz: float) -> None:
    """A component the record cannot carry is not measured at whatever it aliased to."""
    rate_hz = 250.0
    check = check_powerline(
        _record(((50.0, 0.5),), rate_hz, 8.0, seed=29), rate_hz, candidates=(50.0,)
    )
    assert check.best.harmonics_hz == (50.0, 100.0)
    assert check.detected_hz == 50.0
    assert check_powerline(
        _record(((50.0, 0.5),), sample_rate_hz, 4.0, seed=29), sample_rate_hz
    ).best.harmonics_hz == (50.0, 100.0, 150.0)


def test_notching_the_frequency_it_finds_removes_the_line_it_found(
    sample_rate_hz: float,
) -> None:
    """The check and the notch have to agree, or one of them measures the wrong thing."""
    record = _record(((50.0, 0.5), (100.0, 0.25), (150.0, 0.125)), sample_rate_hz, 4.0, seed=31)
    found = check_powerline(record, sample_rate_hz)
    assert found.detected_hz == 50.0

    cleaned = apply_causal(design_powerline_notch(sample_rate_hz, found.detected_hz), record)
    settled = cleaned[round(0.5 * sample_rate_hz) :]
    assert check_powerline(settled, sample_rate_hz).detected_hz is None


def test_the_mains_frequency_of_a_generated_record_is_recovered(sample_rate_hz: float) -> None:
    """On a signal with an electromyogram spectrum, not only on noise plus a sine.

    The generator states its mains frequency and the surface spectrum peaks between
    50 Hz and 150 Hz, which is where the lines are, so this is the case a floor taken
    over the whole record rather than beside the line would get wrong. The interference
    is at half the amplitude of the signal, which is the size a notch exists for.
    """
    for fundamental_hz in (50.0, 60.0):
        trace = generate(
            GenerationSpec(
                sampling=SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=4.0),
                profile=ContractionProfile.single(0.5, 3.5, 0.6),
                noise=NoiseSpec(snr_db=10.0),
                powerline=PowerlineSpec(fundamental_hz=fundamental_hz, amplitude_ratio=0.5),
            ),
            np.random.default_rng(37),
        )
        check = check_powerline(trace.signal, sample_rate_hz)
        assert check.detected_hz == fundamental_hz
        assert check.best.harmonic_excess_db[0] == check.best.excess_db


def test_a_record_too_short_for_the_requested_resolution_is_refused(
    sample_rate_hz: float,
) -> None:
    """Welch shortens its segment to fit the record, which would coarsen the grid.

    A 0.4 s record resolves 2.5 Hz, at which the line window of two bins either side of
    50 Hz reaches 60 Hz and the two candidates stop being distinguishable. Silently
    returning a verdict computed on that grid is the failure this guard exists to stop.
    """
    with pytest.raises(ValueError, match="coarser than the requested"):
        check_powerline(_record(((50.0, 0.5),), sample_rate_hz, 0.4, seed=41), sample_rate_hz)


def test_a_record_with_no_power_cannot_be_checked(sample_rate_hz: float) -> None:
    """A ratio against a floor of zero is not a large excess, it is undefined."""
    with pytest.raises(ValueError, match="no power"):
        check_powerline(np.zeros(4000, dtype=np.float64), sample_rate_hz)


def test_candidates_too_close_together_leave_no_floor_to_measure_against() -> None:
    """Every bin beside 50 Hz belongs to another candidate's line window here.

    The median of an empty selection is not a number, and a floor that is not a number
    would propagate into the excess and out into the verdict.
    """
    rate_hz = 112.0
    with pytest.raises(ValueError, match="no bins beside"):
        check_powerline(
            _record((), rate_hz, 8.0, seed=43),
            rate_hz,
            candidates=(42.0, 46.0, 50.0, 54.0),
            n_harmonics=1,
        )


def test_the_settings_are_validated(sample_rate_hz: float) -> None:
    """Each of these would produce a number computed under conditions nobody chose."""
    record = _record(((50.0, 0.5),), sample_rate_hz, 4.0, seed=47)
    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        check_powerline(record, 0.0)
    with pytest.raises(ValueError, match="at least one candidate"):
        check_powerline(record, sample_rate_hz, candidates=())
    with pytest.raises(ValueError, match="n_harmonics must be at least 1"):
        check_powerline(record, sample_rate_hz, n_harmonics=0)
    with pytest.raises(ValueError, match="resolution_hz must be positive"):
        check_powerline(record, sample_rate_hz, resolution_hz=0.0)
    with pytest.raises(ValueError, match="must lie strictly between"):
        check_powerline(record, sample_rate_hz, candidates=(50.0, sample_rate_hz))


def test_the_table_lists_every_candidate_and_the_verdict(sample_rate_hz: float) -> None:
    """A check that cannot be read as a table cannot be put in a report."""
    check = check_powerline(_record(((50.0, 0.5),), sample_rate_hz, 4.0, seed=53), sample_rate_hz)
    table = format_powerline_table(check)
    lines = table.splitlines()

    assert lines[0].startswith("| Candidate (Hz) |")
    assert len(lines) == 2 + len(check.candidates) + 2
    assert lines[2].startswith("| 50 | 3 |")
    assert lines[3].startswith("| 60 | 3 |")
    assert lines[-1] == f"{check.reason}."
