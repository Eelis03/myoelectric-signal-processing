"""Tests for the interface through which a real recording is substituted.

No dataset is shipped, so what is tested here is that the interface accepts the two
formats a real dataset is usually exported to and that everything downstream sees the
same object whether the data came from a file or from the generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myoelectric.algorithm.features_time import root_mean_square
from myoelectric.model.contraction import ContractionProfile
from myoelectric.model.noise import NoiseSpec
from myoelectric.model.sampling import SamplingSpec
from myoelectric.pipeline.generation import GenerationSpec, generate
from myoelectric.pipeline.loaders import (
    CsvRecordingLoader,
    EmgRecording,
    NpzRecordingLoader,
    RecordingLoader,
    recording_from_trace,
)


def _samples() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.standard_normal((512, 3))


def test_npz_loader_round_trips_a_recording(tmp_path: Path) -> None:
    """An archive written with the documented keys loads back unchanged."""
    samples = _samples()
    path = tmp_path / "recording.npz"
    np.savez(
        path,
        samples=samples,
        sample_rate_hz=np.array([1000.0]),
        channel_names=np.array(["flexor", "extensor", "reference"]),
        onset_indices=np.array([120, 300]),
    )
    loader = NpzRecordingLoader(label="wrist flexion")
    assert isinstance(loader, RecordingLoader)

    recording = loader.load(path)
    assert recording.samples == pytest.approx(samples, rel=1e-12)
    assert recording.sample_rate_hz == 1000.0
    assert recording.channel_names == ("flexor", "extensor", "reference")
    assert recording.onset_indices == (120, 300)
    assert recording.label == "wrist flexion"
    assert recording.n_channels == 3
    assert recording.n_samples == 512
    assert recording.duration_s == pytest.approx(0.512, rel=1e-12)


def test_npz_loader_supplies_defaults_for_optional_keys(tmp_path: Path) -> None:
    """A minimal archive still loads, with generated channel names and no annotations."""
    path = tmp_path / "minimal.npz"
    np.savez(path, samples=_samples(), sample_rate_hz=np.array([2000.0]))
    recording = NpzRecordingLoader().load(path)
    assert recording.channel_names == ("ch0", "ch1", "ch2")
    assert recording.onset_indices == ()


def test_csv_loader_reads_a_delimited_export(tmp_path: Path) -> None:
    """A comma separated export with a header loads with its channel names."""
    path = tmp_path / "recording.csv"
    path.write_text("flexor,extensor\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")
    loader = CsvRecordingLoader(sample_rate_hz=1000.0, label="grip")
    assert isinstance(loader, RecordingLoader)

    recording = loader.load(path)
    assert recording.channel_names == ("flexor", "extensor")
    assert recording.channel("extensor") == pytest.approx(np.array([2.0, 4.0, 6.0]), rel=1e-12)
    assert recording.channel(0) == pytest.approx(np.array([1.0, 3.0, 5.0]), rel=1e-12)
    assert recording.label == "grip"


def test_csv_loader_without_a_header_generates_names(tmp_path: Path) -> None:
    """An export with no header still loads."""
    path = tmp_path / "headerless.csv"
    path.write_text("1.0;2.0\n3.0;4.0\n", encoding="utf-8")
    recording = CsvRecordingLoader(
        sample_rate_hz=1000.0, delimiter=";", has_header=False
    ).load(path)
    assert recording.channel_names == ("ch0", "ch1")
    assert recording.n_samples == 2


def test_csv_loader_rejects_an_empty_file(tmp_path: Path) -> None:
    """An empty export is an error rather than an empty recording."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        CsvRecordingLoader(sample_rate_hz=1000.0).load(path)


def test_generated_trace_wraps_as_a_recording(sample_rate_hz: float) -> None:
    """The synthetic path and the real data path present the same object downstream."""
    trace = generate(
        GenerationSpec(
            sampling=SamplingSpec(sample_rate_hz=sample_rate_hz, duration_s=1.0),
            profile=ContractionProfile.single(0.3, 0.9, 0.6),
            noise=NoiseSpec(snr_db=20.0),
        ),
        np.random.default_rng(23),
    )
    recording = recording_from_trace(trace)
    assert recording.n_channels == 1
    assert recording.onset_indices == trace.onset_indices
    assert recording.sampling.sample_rate_hz == sample_rate_hz
    assert root_mean_square(recording.channel(0)) == pytest.approx(
        root_mean_square(trace.signal), rel=1e-12
    )


def test_recording_validates_its_shape() -> None:
    """A channel count that disagrees with the channel names is rejected."""
    with pytest.raises(ValueError):
        EmgRecording(samples=np.zeros((10, 2)), sample_rate_hz=1000.0, channel_names=("only",))
    with pytest.raises(ValueError):
        EmgRecording(samples=np.zeros(10), sample_rate_hz=1000.0, channel_names=("only",))
    with pytest.raises(ValueError):
        EmgRecording(samples=np.zeros((10, 1)), sample_rate_hz=0.0, channel_names=("only",))
