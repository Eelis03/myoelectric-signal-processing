"""Filtering, onset detection, and features for myoelectric signals.

The package is organised in four library layers plus the example scripts:

``myoelectric.model``
    Value objects: sampling geometry, motor unit pool, contraction profile, noise
    specifications. No input or output, no computation beyond validation.

``myoelectric.algorithm``
    Filters, onset detectors behind a protocol, and the time and frequency domain
    feature functions. Deterministic functions of their arguments.

``myoelectric.pipeline``
    Synthetic signal generation, the detector evaluation sweep, the fatigue protocol,
    the amplitude estimator latency study, and the loading interface for real
    recordings. Each returns a structured trace.

``myoelectric.analysis``
    Detector metrics, fatigue statistics, tables, and figures, all computed from
    traces.

``myoelectric.analysis.figures`` is not imported by ``myoelectric.analysis`` because it
pulls in matplotlib, which the rest of the library does not need. Import it directly.

Results reported by this project are computed on synthetic signals. See the README
overview and ``docs/design-notes.md`` for what that does and does not establish.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
