"""
MZI Sensor Calibration — Physics-based calibration of IMTL Mach-Zehnder
interferometric dielectric sensors using Method of Moments and Green's Function.

Modules
-------
greens      Spectral-domain Green's function for multilayer microstrip
mom         Galerkin Method of Moments solver for propagation constant
mzi         Mach-Zehnder interferometer S-parameter network model
calibration Two-anchor affine calibration framework
inversion   Physics-based multi-start bounded inversion engine
materials   Dielectric material property database
plotting    Visualization utilities
"""

__version__ = "1.0.0"
__author__ = "Amirreza Piri"

from .greens import SpectralGreenFunction, LayeredMedium, DielectricLayer
from .mom import GalerkinMoM, MoMConfig
from .mzi import MZIModel
from .calibration import AffineCalibration
from .inversion import PhysicsInversion, InversionResult
from .materials import (
    ANALYTES, AIR, WATER, VALIDATION_ANALYTES,
    SensorGeometry, DEFAULT_GEOMETRY,
)
