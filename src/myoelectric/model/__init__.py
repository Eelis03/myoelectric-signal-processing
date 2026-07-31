"""Model layer: value objects describing signals, motor units, and contractions.

This layer holds no input or output logic and performs no plotting. It defines the
parameters that the algorithm and pipeline layers consume.
"""

from myoelectric.model.contraction import ContractionEvent, ContractionProfile
from myoelectric.model.motor_unit import MotorUnit, MotorUnitPool, MotorUnitPoolSpec
from myoelectric.model.noise import MotionArtefactSpec, NoiseSpec, PowerlineSpec
from myoelectric.model.sampling import SamplingSpec

__all__ = [
    "ContractionEvent",
    "ContractionProfile",
    "MotionArtefactSpec",
    "MotorUnit",
    "MotorUnitPool",
    "MotorUnitPoolSpec",
    "NoiseSpec",
    "PowerlineSpec",
    "SamplingSpec",
]
