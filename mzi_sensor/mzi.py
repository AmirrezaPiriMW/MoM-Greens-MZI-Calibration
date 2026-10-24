"""
Mach-Zehnder Interferometer (MZI) S-parameter Network Model
============================================================

Models the out-of-phase IMTL MZI topology used in the dielectric sensor.
The MZI splits the input signal into two arms (reference and sensing),
each a microstrip transmission line with its own propagation constant.
The outputs combine out-of-phase (180° hybrid), producing:

    S₂₁ = ½ [e^{−γ_ref · L_ref}  −  e^{−γ_sense · L_sense}]

where γ = α + jβ is the complex propagation constant.

The deep null in |S₂₁| occurs when both arms have equal amplitude and
phase, making the interferometer exquisitely sensitive to changes in the
sensing arm's dielectric environment.

References
----------
[1] A. Piri et al., "IMTL Mach-Zehnder Interferometric Dielectric
    Sensor," University of Pavia, 2024.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MZIModel:
    """Out-of-phase Mach-Zehnder interferometer model.

    Parameters
    ----------
    L_ref : float
        Reference arm length [m].
    L_sense : float
        Sensing arm length [m].
    gamma_ref : complex
        Propagation constant of the reference arm γ = α + jβ [1/m].
    """
    L_ref: float
    L_sense: float
    gamma_ref: complex

    def s21(self, gamma_sense: complex) -> complex:
        """Compute S₂₁ for the out-of-phase MZI topology.

        S₂₁ = ½ [exp(−γ_ref · L_ref) − exp(−γ_sense · L_sense)]

        Parameters
        ----------
        gamma_sense : complex
            Propagation constant of the sensing arm γ = α + jβ.

        Returns
        -------
        S21 : complex
            Complex transmission coefficient.
        """
        ref_term = np.exp(-self.gamma_ref * self.L_ref)
        sense_term = np.exp(-gamma_sense * self.L_sense)
        return 0.5 * (ref_term - sense_term)

    def s21_db_deg(self, gamma_sense: complex) -> tuple:
        """Compute S₂₁ in dB and degrees.

        Returns
        -------
        S21_dB : float
            |S₂₁| in decibels.
        S21_deg : float
            ∠S₂₁ in degrees.
        """
        s = self.s21(gamma_sense)
        s21_dB = 20.0 * np.log10(np.abs(s) + 1e-30)
        s21_deg = np.degrees(np.angle(s))
        return float(s21_dB), float(s21_deg)

    def observable(self, gamma_sense: complex) -> np.ndarray:
        """Return the observable vector [Re(S₂₁), Im(S₂₁)].

        This is the quantity used in the calibration framework.
        """
        s = self.s21(gamma_sense)
        return np.array([np.real(s), np.imag(s)])

    @staticmethod
    def db_deg_to_complex(s21_dB: float, s21_deg: float) -> complex:
        """Convert S₂₁ from dB/degrees to complex."""
        mag = 10.0 ** (s21_dB / 20.0)
        phase = np.radians(s21_deg)
        return mag * np.exp(1j * phase)

    @staticmethod
    def complex_to_db_deg(s21: complex) -> tuple:
        """Convert complex S₂₁ to (dB, degrees)."""
        dB = 20.0 * np.log10(np.abs(s21) + 1e-30)
        deg = np.degrees(np.angle(s21))
        return float(dB), float(deg)
