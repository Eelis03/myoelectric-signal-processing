"""Loading interface for real surface electromyography recordings.

This project ships no dataset and downloads none. Every number it reports is produced
from the synthetic generator in :mod:`myoelectric.pipeline.generation`. The interface
here exists so that a reader with access to a real recording can substitute it without
modifying any algorithm code: everything downstream of :class:`EmgRecording` consumes an
array and a sample rate and does not care where they came from.

Public datasets suitable for a real evaluation, none of which is redistributed here:

Ninapro
    Multi channel surface recordings during a large gesture set, with synchronised
    stimulus labels that provide gesture boundaries. Atzori et al. (2014). Available
    from https://ninapro.hevs.ch after registration. Export one channel of one
    repetition to a two column comma separated file and load it with
    :class:`CsvRecordingLoader`, or to a compressed archive and load it with
    :class:`NpzRecordingLoader`.

putEMG
    Twenty four channel surface recordings of eight hand gestures from forty four
    subjects, distributed with per sample gesture labels. Kaczmarek et al. (2019).
    Available from https://biolab.put.poznan.pl/putemg-dataset/.

PhysioNet examples of electromyograms
    Short single channel needle recordings from a healthy subject, a subject with
    myopathy and a subject with neuropathy. Useful as a sanity check on filter and
    feature code rather than as a control dataset. Goldberger et al. (2000).
    Available from https://physionet.org/content/emgdb/1.0.0/.

When substituting real data, three things change and must be checked. The sample rate is
usually 1000 Hz or 2000 Hz rather than the 2000 Hz assumed here, so every filter must be
redesigned at the recording rate rather than reused. The mains frequency is 50 Hz or
60 Hz depending on where the recording was made, and
:func:`myoelectric.analysis.powerline_check.check_powerline` reads it off a channel of
the recording when the dataset does not state it. Ground truth onsets in a labelled
dataset mark the instant a subject was cued or the instant an experimenter annotated,
neither of which is the instant the first motor unit discharged, so a timing bias
measured against them is not comparable with the bias reported here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import SignalTrace

__all__ = [
    "CsvRecordingLoader",
    "EmgRecording",
    "NpzRecordingLoader",
    "RecordingLoader",
    "recording_from_trace",
]


@dataclass(frozen=True, slots=True, eq=False)
class EmgRecording:
    """One multi channel surface electromyography recording.

    Attributes:
        samples: Array of shape ``(n_samples, n_channels)`` in the units of the
            recording, usually millivolts at the amplifier input.
        sample_rate_hz: Sample rate.
        channel_names: One name per column.
        onset_indices: Ground truth onset sample indices where they are known. Empty
            when the recording carries no annotation.
        label: Gesture or condition label.
        source: Where the recording came from, carried through into reports.
    """

    samples: NDArray[np.float64]
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    onset_indices: tuple[int, ...] = ()
    label: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError(f"samples must be two dimensional, got shape {self.samples.shape}")
        if self.samples.shape[1] != len(self.channel_names):
            raise ValueError(
                f"got {self.samples.shape[1]} channels but {len(self.channel_names)} channel names"
            )
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        """Length of the recording in seconds."""
        return self.n_samples / self.sample_rate_hz

    @property
    def sampling(self) -> SamplingSpec:
        """Sampling geometry, so that filters can be designed at the recording rate."""
        return SamplingSpec(sample_rate_hz=self.sample_rate_hz, duration_s=self.duration_s)

    def channel(self, name: str | int) -> NDArray[np.float64]:
        """One channel as a contiguous one dimensional array."""
        index = name if isinstance(name, int) else self.channel_names.index(name)
        return np.ascontiguousarray(self.samples[:, index], dtype=np.float64)


@runtime_checkable
class RecordingLoader(Protocol):
    """Common interface for anything that produces an :class:`EmgRecording`."""

    def load(self, path: Path) -> EmgRecording:
        """Read one recording from ``path``."""
        ...


@dataclass(frozen=True, slots=True)
class NpzRecordingLoader:
    """Loader for a compressed numpy archive.

    The archive must contain ``samples`` of shape ``(n_samples, n_channels)`` and a
    scalar ``sample_rate_hz``. It may contain ``channel_names`` as an array of strings
    and ``onset_indices`` as an integer array. This is the format to convert a dataset
    into once, so that the conversion is done in one place and everything downstream
    reads the same thing.
    """

    label: str = ""

    def load(self, path: Path) -> EmgRecording:
        """Read one recording from ``path``."""
        with np.load(path, allow_pickle=False) as archive:
            samples = np.atleast_2d(np.asarray(archive["samples"], dtype=np.float64))
            if samples.shape[0] == 1 and samples.shape[1] > 1:
                samples = samples.T
            sample_rate = float(np.asarray(archive["sample_rate_hz"]).reshape(-1)[0])
            names = (
                tuple(str(name) for name in archive["channel_names"])
                if "channel_names" in archive
                else tuple(f"ch{i}" for i in range(samples.shape[1]))
            )
            onsets = (
                tuple(int(value) for value in archive["onset_indices"])
                if "onset_indices" in archive
                else ()
            )
        return EmgRecording(
            samples=samples,
            sample_rate_hz=sample_rate,
            channel_names=names,
            onset_indices=onsets,
            label=self.label,
            source=str(path),
        )


@dataclass(frozen=True, slots=True)
class CsvRecordingLoader:
    """Loader for a delimited text export with one column per channel.

    Args:
        sample_rate_hz: Sample rate of the export, which a plain text file cannot carry.
        delimiter: Column delimiter.
        has_header: Whether the first row holds channel names.
        label: Gesture or condition label attached to the resulting recording.
    """

    sample_rate_hz: float
    delimiter: str = ","
    has_header: bool = True
    label: str = ""

    def load(self, path: Path) -> EmgRecording:
        """Read one recording from ``path``."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=self.delimiter)
            rows = [row for row in reader if row]
        if not rows:
            raise ValueError(f"{path} contains no data")
        if self.has_header:
            names = tuple(name.strip() for name in rows[0])
            body = rows[1:]
        else:
            names = tuple(f"ch{i}" for i in range(len(rows[0])))
            body = rows
        if not body:
            raise ValueError(f"{path} contains a header but no samples")
        samples = np.asarray([[float(value) for value in row] for row in body], dtype=np.float64)
        return EmgRecording(
            samples=samples,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=names,
            label=self.label,
            source=str(path),
        )


def recording_from_trace(trace: SignalTrace, label: str = "synthetic") -> EmgRecording:
    """Wrap a generated trace as an :class:`EmgRecording`.

    This is the adapter that makes the synthetic path and the real data path identical
    from the point of view of every analysis in this library.
    """
    return EmgRecording(
        samples=trace.signal.reshape(-1, 1),
        sample_rate_hz=trace.sampling.sample_rate_hz,
        channel_names=("synthetic",),
        onset_indices=trace.onset_indices,
        label=label,
        source="myoelectric.pipeline.generation",
    )
