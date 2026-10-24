"""
Two-Anchor Affine Calibration Framework
========================================

Corrects the systematic offset between MoM-simulated and measured
observables (S₂₁) using a two-point affine transformation:

    obs_meas = a · obs_model + b

The two anchor substances are Air (ε' ≈ 1) and Water (ε' ≈ 77.6),
spanning the full permittivity range of the sensor.

The calibration operates on the complex S₂₁ decomposed into real and
imaginary parts, applying separate affine maps to each component.

After calibration, the MoM forward model produces S₂₁ predictions that
match measured data, enabling physics-based inversion for unknown analytes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class AffineCalibration:
    """Two-anchor affine calibration for MZI observables.

    Computes and applies the affine map  y = a·x + b  that matches
    the MoM model's S₂₁ to measured S₂₁ at two anchor points.

    Parameters
    ----------
    obs_model_anchor1 : ndarray, shape (2,)
        Model observable [Re(S₂₁), Im(S₂₁)] at anchor 1 (Air).
    obs_meas_anchor1 : ndarray, shape (2,)
        Measured observable at anchor 1.
    obs_model_anchor2 : ndarray, shape (2,)
        Model observable at anchor 2 (Water).
    obs_meas_anchor2 : ndarray, shape (2,)
        Measured observable at anchor 2.
    """
    obs_model_anchor1: np.ndarray
    obs_meas_anchor1: np.ndarray
    obs_model_anchor2: np.ndarray
    obs_meas_anchor2: np.ndarray

    # Calibration coefficients [a_re, a_im] and [b_re, b_im]
    a: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self):
        """Compute affine coefficients from the two anchors."""
        # For each component (Re, Im):
        # meas = a * model + b
        # Using two points: a = (meas2 - meas1) / (model2 - model1)
        #                    b = meas1 - a * model1
        delta_model = self.obs_model_anchor2 - self.obs_model_anchor1
        delta_meas = self.obs_meas_anchor2 - self.obs_meas_anchor1

        # Guard against division by zero
        safe = np.abs(delta_model) > 1e-30
        self.a = np.where(safe, delta_meas / delta_model, 1.0)
        self.b = self.obs_meas_anchor1 - self.a * self.obs_model_anchor1

    def apply(self, obs_model: np.ndarray) -> np.ndarray:
        """Apply calibration to a model observable.

        Parameters
        ----------
        obs_model : ndarray, shape (2,)
            Raw model observable [Re(S₂₁), Im(S₂₁)].

        Returns
        -------
        obs_cal : ndarray, shape (2,)
            Calibrated observable matching the measurement frame.
        """
        return self.a * obs_model + self.b

    def apply_inverse(self, obs_meas: np.ndarray) -> np.ndarray:
        """Inverse calibration: map measured back to model frame.

        Parameters
        ----------
        obs_meas : ndarray, shape (2,)
            Measured observable.

        Returns
        -------
        obs_model : ndarray, shape (2,)
            Observable in the model frame.
        """
        safe = np.abs(self.a) > 1e-30
        return np.where(safe, (obs_meas - self.b) / self.a, obs_meas)

    def residual(self, obs_model: np.ndarray,
                 obs_meas: np.ndarray) -> float:
        """Euclidean distance between calibrated model and measurement.

        Parameters
        ----------
        obs_model : ndarray, shape (2,)
            Raw model observable.
        obs_meas : ndarray, shape (2,)
            Measured observable.

        Returns
        -------
        residual : float
            ‖cal(model) − meas‖₂
        """
        cal = self.apply(obs_model)
        return float(np.linalg.norm(cal - obs_meas))

    def summary(self) -> str:
        """Human-readable calibration summary."""
        lines = [
            "Affine Calibration Coefficients",
            "─" * 40,
            f"  Re: y = {self.a[0]:.6f} · x + {self.b[0]:.6f}",
            f"  Im: y = {self.a[1]:.6f} · x + {self.b[1]:.6f}",
        ]
        return "\n".join(lines)
