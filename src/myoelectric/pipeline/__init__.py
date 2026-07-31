"""Pipeline layer: signal generation, evaluation sweeps, and data loading.

Each entry point in this layer takes a specification value object and returns a
structured trace. Traces carry every intermediate quantity that the analysis layer
needs, so the analysis layer never has to rerun a pipeline to obtain a number.
"""

from myoelectric.pipeline.detection_sweep import (
    DetectorSweepTrace,
    SweepSpec,
    TrialOutcome,
    run_detector_sweep,
)
from myoelectric.pipeline.fatigue import FatigueSpec, FatigueTrace, run_fatigue_protocol
from myoelectric.pipeline.generation import GenerationSpec, SignalTrace, generate
from myoelectric.pipeline.latency import LatencySpec, LatencyTrace, run_latency_study
from myoelectric.pipeline.loaders import (
    CsvRecordingLoader,
    EmgRecording,
    NpzRecordingLoader,
    RecordingLoader,
    recording_from_trace,
)

__all__ = [
    "CsvRecordingLoader",
    "DetectorSweepTrace",
    "EmgRecording",
    "FatigueSpec",
    "FatigueTrace",
    "GenerationSpec",
    "LatencySpec",
    "LatencyTrace",
    "NpzRecordingLoader",
    "RecordingLoader",
    "SignalTrace",
    "SweepSpec",
    "TrialOutcome",
    "generate",
    "recording_from_trace",
    "run_detector_sweep",
    "run_fatigue_protocol",
    "run_latency_study",
]
