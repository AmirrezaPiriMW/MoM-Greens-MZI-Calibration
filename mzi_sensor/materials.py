"""
Dielectric Material Property Database
======================================

Reference permittivity data for the 9 calibration analytes measured
on the IMTL MZI sensor at 2.45 GHz, 29 ± 1 °C.

All values are from independent reference measurements (literature
and/or Agilent 85070E dielectric probe).

The measured S₂₁ data (dB, degrees) correspond to the out-of-phase
MZI topology response.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Analyte:
    """A single calibration analyte."""
    name: str
    eps_prime: float       # ε' (real part of relative permittivity)
    tan_delta: float       # loss tangent
    s21_dB: float          # measured |S₂₁| [dB]
    s21_deg: float         # measured ∠S₂₁ [degrees]

    @property
    def eps_double_prime(self) -> float:
        """Imaginary part: ε″ = ε' · tan δ."""
        return self.eps_prime * self.tan_delta

    @property
    def eps_complex(self) -> complex:
        """Complex permittivity: ε = ε' − jε″."""
        return self.eps_prime * (1.0 - 1j * self.tan_delta)

    @property
    def s21_complex(self) -> complex:
        """Complex S₂₁ from measured dB/degrees."""
        mag = 10.0 ** (self.s21_dB / 20.0)
        phase = np.radians(self.s21_deg)
        return mag * np.exp(1j * phase)


# ═══════════════════════════════════════════════════════════════════
# Canonical calibration dataset — 9 analytes at 2.45 GHz, 29 ± 1 °C
# ═══════════════════════════════════════════════════════════════════

ANALYTES: List[Analyte] = [
    Analyte("Air",      1.003,  0.000, -7.605070,  -54.380024),
    Analyte("IPA",      5.294,  0.757, -16.890303, -33.389763),
    Analyte("80% IPA",  16.680, 0.590, -27.185883,  -5.310281),
    Analyte("90% EtOH", 17.844, 0.680, -28.216131,  13.559215),
    Analyte("Acetone",  21.925, 0.034, -39.647438, -27.719151),
    Analyte("60% IPA",  31.999, 0.396, -31.622412,  40.923534),
    Analyte("40% IPA",  48.406, 0.285, -30.121349,  66.664047),
    Analyte("20% IPA",  64.667, 0.189, -30.226261,  73.713295),
    Analyte("Water",    77.587, 0.106, -29.293024,  78.048912),
]

# Air and Water are the calibration anchors
AIR = ANALYTES[0]
WATER = ANALYTES[-1]

# Validation analytes (everything except Air and Water)
VALIDATION_ANALYTES = ANALYTES[1:-1]


def get_analyte(name: str) -> Analyte:
    """Look up an analyte by name (case-insensitive)."""
    for a in ANALYTES:
        if a.name.lower() == name.lower():
            return a
    raise KeyError(f"Unknown analyte: {name}")


def eps_prime_array() -> np.ndarray:
    """Array of reference ε' values."""
    return np.array([a.eps_prime for a in ANALYTES])


def tan_delta_array() -> np.ndarray:
    """Array of reference tan δ values."""
    return np.array([a.tan_delta for a in ANALYTES])


def s21_dB_array() -> np.ndarray:
    """Array of measured |S₂₁| in dB."""
    return np.array([a.s21_dB for a in ANALYTES])


def s21_deg_array() -> np.ndarray:
    """Array of measured ∠S₂₁ in degrees."""
    return np.array([a.s21_deg for a in ANALYTES])


# ═══════════════════════════════════════════════════════════════════
# Sensor geometry (IMTL microstrip at 2.45 GHz)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SensorGeometry:
    """Physical dimensions of the IMTL MZI sensor."""
    frequency: float = 2.45e9           # [Hz]
    strip_width: float = 3.0e-3         # [m]  conductor width
    substrate_thickness: float = 1.6e-3 # [m]  FR-4 substrate
    substrate_eps_r: float = 4.4        # FR-4 permittivity
    substrate_tand: float = 0.02        # FR-4 loss tangent
    channel_height: float = 2.0e-3      # [m]  analyte channel depth
    L_ref: float = 40.0e-3              # [m]  reference arm length
    L_sense: float = 40.0e-3            # [m]  sensing arm length


DEFAULT_GEOMETRY = SensorGeometry()
