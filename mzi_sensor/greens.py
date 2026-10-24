"""
Spectral-Domain Green's Function for Multilayer Microstrip Structures
=====================================================================

Computes the dyadic Green's function in the spectral domain for a microstrip
transmission line embedded in a multilayer dielectric structure. The formulation
follows the immittance approach of Itoh (1989), decomposing the fields into
TM-to-y (e-type) and TE-to-y (h-type) partial waves and cascading layer
admittances via the transmission-line analogy.

Structure (bottom to top)
-------------------------
    Layer 0 : PEC ground plane
    Layer 1 : Substrate  (εr_sub, μr=1, thickness h_sub)
    ──────── : Strip conductor at y = h_sub, width W
    Layer 2 : Analyte / superstrate (εr_a = ε' − jε″, thickness h_ch)
    Layer 3 : Air half-space (εr = 1)

References
----------
[1] T. Itoh, "Spectral Domain Immittance Approach for Dispersion
    Characteristics of Generalized Printed Transmission Lines," IEEE
    Trans. MTT, vol. 28, no. 7, pp. 733–736, Jul. 1980.
[2] D. M. Pozar, "Microwave Engineering," 4th ed., Wiley, 2012.
[3] R. E. Collin, "Foundations for Microwave Engineering," 2nd ed.,
    IEEE Press, 2001.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# Physical constants
MU_0 = 4.0e-7 * np.pi          # [H/m]  vacuum permeability
EPS_0 = 8.854187817e-12         # [F/m]  vacuum permittivity
C_0 = 1.0 / np.sqrt(MU_0 * EPS_0)  # [m/s]  speed of light
ETA_0 = np.sqrt(MU_0 / EPS_0)  # [Ω]    free-space impedance


@dataclass
class DielectricLayer:
    """A single dielectric layer in the multilayer stack.

    Parameters
    ----------
    eps_r : complex
        Relative permittivity (real part ε', imaginary part −ε″ for loss).
        Convention: eps_r = eps_prime - 1j * eps_prime * tan_delta.
    thickness : float
        Layer thickness in metres.  Use ``np.inf`` for a semi-infinite layer.
    name : str
        Human-readable label.
    """
    eps_r: complex
    thickness: float
    name: str = ""

    @classmethod
    def from_eps_tand(cls, eps_prime: float, tan_delta: float,
                      thickness: float, name: str = "") -> "DielectricLayer":
        """Construct from real permittivity and loss tangent."""
        eps_r = eps_prime * (1.0 - 1j * tan_delta)
        return cls(eps_r=eps_r, thickness=thickness, name=name)


@dataclass
class LayeredMedium:
    """Defines the complete multilayer environment around the strip.

    The strip sits at the interface between ``substrate`` (below, terminated
    by PEC ground) and ``superstrate`` layers (above, terminated by a
    semi-infinite air half-space).
    """
    substrate: DielectricLayer
    channel: DielectricLayer     # analyte layer above strip
    frequency: float             # operating frequency [Hz]
    strip_width: float           # conductor width [m]

    # Derived quantities (computed on init)
    omega: float = field(init=False)
    k0: float = field(init=False)

    def __post_init__(self):
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / C_0

    def wavenumber(self, layer: DielectricLayer) -> complex:
        """Wavenumber k_i = ω√(μ₀ε_i) in a layer."""
        return self.k0 * np.sqrt(layer.eps_r + 0j)


class SpectralGreenFunction:
    """Spectral-domain Green's function G̃_zz(αx, β) for the microstrip.

    Given a spectral variable αx (Fourier-conjugate of x, the transverse
    coordinate) and a trial propagation constant β along z, this class
    computes the Green's function kernel that enters the Galerkin MoM
    dispersion equation.

    The formulation decomposes the problem into e-type (TM to y) and
    h-type (TE to y) partial fields and uses transmission-line analogy
    to cascade layer impedances.

    Parameters
    ----------
    medium : LayeredMedium
        The multilayer structure description.
    """

    def __init__(self, medium: LayeredMedium):
        self.med = medium

    # ──────────────────────────────────────────────────────────────────
    # Core: transverse wavenumber in each layer
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _gamma_y(alpha_x: np.ndarray, beta: complex,
                 k_layer: complex) -> np.ndarray:
        """Transverse (y-directed) propagation constant in a layer.

        γ_y = √(αx² + β² − k_i²)

        The branch is chosen so that Re(γ_y) ≥ 0 (proper wave, decaying
        away from the source).
        """
        arg = alpha_x**2 + beta**2 - k_layer**2
        gy = np.sqrt(arg + 0j)
        # Enforce proper branch: Re(γ_y) ≥ 0
        gy = np.where(np.real(gy) < 0, -gy, gy)
        return gy

    # ──────────────────────────────────────────────────────────────────
    # TM (e-type) and TE (h-type) characteristic admittances
    # ──────────────────────────────────────────────────────────────────
    def _Y_e(self, gamma_y: np.ndarray,
             layer: DielectricLayer) -> np.ndarray:
        """TM (e-type) characteristic admittance: Y_e = jωε_i / γ_y."""
        eps_i = layer.eps_r * EPS_0
        return 1j * self.med.omega * eps_i / gamma_y

    def _Y_h(self, gamma_y: np.ndarray) -> np.ndarray:
        """TE (h-type) characteristic admittance: Y_h = γ_y / (jωμ₀)."""
        return gamma_y / (1j * self.med.omega * MU_0)

    # ──────────────────────────────────────────────────────────────────
    # Input admittance cascading via transmission-line formula
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _Y_input(Y0: np.ndarray, gamma_y: np.ndarray,
                 thickness: float, Y_load: np.ndarray) -> np.ndarray:
        """Input admittance of a TL section terminated by Y_load.

        Y_in = Y₀ · (Y_load + Y₀·tanh(γ·d)) / (Y₀ + Y_load·tanh(γ·d))

        For a short-circuited line (PEC ground, Y_load → ∞):
        Y_in = Y₀ · coth(γ·d)
        """
        gd = gamma_y * thickness
        # Clip to avoid overflow in tanh/coth
        gd = np.clip(np.real(gd), -500, 500) + 1j * np.imag(gd)
        th = np.tanh(gd)
        return Y0 * (Y_load + Y0 * th) / (Y0 + Y_load * th)

    @staticmethod
    def _Y_short(Y0: np.ndarray, gamma_y: np.ndarray,
                 thickness: float) -> np.ndarray:
        """Input admittance of a short-circuited TL: Y₀·coth(γ·d)."""
        gd = gamma_y * thickness
        gd = np.clip(np.real(gd), -500, 500) + 1j * np.imag(gd)
        return Y0 / np.tanh(gd)

    # ──────────────────────────────────────────────────────────────────
    # Admittances looking down (into substrate) and up (into superstrate)
    # ──────────────────────────────────────────────────────────────────
    def admittance_below(self, alpha_x: np.ndarray,
                         beta: complex) -> Tuple[np.ndarray, np.ndarray]:
        """TM and TE admittances looking down into substrate + PEC ground.

        Returns (Y_e_down, Y_h_down).
        """
        sub = self.med.substrate
        k_sub = self.med.wavenumber(sub)
        gy_sub = self._gamma_y(alpha_x, beta, k_sub)

        Y_e0 = self._Y_e(gy_sub, sub)
        Y_h0 = self._Y_h(gy_sub)

        # Short-circuited by PEC ground
        Y_e_down = self._Y_short(Y_e0, gy_sub, sub.thickness)
        Y_h_down = self._Y_short(Y_h0, gy_sub, sub.thickness)

        return Y_e_down, Y_h_down

    def admittance_above(self, alpha_x: np.ndarray,
                         beta: complex) -> Tuple[np.ndarray, np.ndarray]:
        """TM and TE admittances looking up into analyte layer + air.

        Returns (Y_e_up, Y_h_up).
        """
        ch = self.med.channel
        k_ch = self.med.wavenumber(ch)
        gy_ch = self._gamma_y(alpha_x, beta, k_ch)

        # Air half-space
        air = DielectricLayer(eps_r=1.0, thickness=np.inf, name="air")
        k_air = self.med.wavenumber(air)
        gy_air = self._gamma_y(alpha_x, beta, k_air)
        Y_e_air = self._Y_e(gy_air, air)
        Y_h_air = self._Y_h(gy_air)

        # Cascade through analyte channel
        Y_e_ch = self._Y_e(gy_ch, ch)
        Y_h_ch = self._Y_h(gy_ch)

        Y_e_up = self._Y_input(Y_e_ch, gy_ch, ch.thickness, Y_e_air)
        Y_h_up = self._Y_input(Y_h_ch, gy_ch, ch.thickness, Y_h_air)

        return Y_e_up, Y_h_up

    # ──────────────────────────────────────────────────────────────────
    # Green's function kernel
    # ──────────────────────────────────────────────────────────────────
    def kernel(self, alpha_x: np.ndarray, beta: complex) -> np.ndarray:
        """Spectral-domain Green's function G̃_zz(αx, β).

        The zz-component of the dyadic Green's function in the spectral
        domain relates the z-directed electric field to z-directed surface
        current at the strip interface:

            G̃_zz = [β²·F_e + αx²·F_h] / (αx² + β²)

        where F_e = 1/(Y_e_down + Y_e_up) and F_h = 1/(Y_h_down + Y_h_up)
        are the TM and TE spectral-domain response functions.

        Parameters
        ----------
        alpha_x : array_like
            Spectral variable (Fourier-conjugate of transverse x).
        beta : complex
            Trial propagation constant along z.

        Returns
        -------
        G : ndarray (complex)
            Green's function evaluated at each αx.
        """
        alpha_x = np.asarray(alpha_x, dtype=complex)
        Y_e_down, Y_h_down = self.admittance_below(alpha_x, beta)
        Y_e_up, Y_h_up = self.admittance_above(alpha_x, beta)

        # Immittance response functions
        F_e = 1.0 / (Y_e_down + Y_e_up)
        F_h = 1.0 / (Y_h_down + Y_h_up)

        # Spectral Green's function
        alpha_sq = alpha_x**2
        beta_sq = beta**2
        denom = alpha_sq + beta_sq

        # Avoid division by zero at αx = β = 0
        safe = np.abs(denom) > 1e-30
        G = np.zeros_like(alpha_x, dtype=complex)
        G[safe] = (beta_sq * F_e[safe] + alpha_sq[safe] * F_h[safe]) / denom[safe]

        return G

    def kernel_decomposed(self, alpha_x: np.ndarray,
                          beta: complex) -> Tuple[np.ndarray, np.ndarray]:
        """Return TM and TE response functions F_e and F_h separately.

        Useful for diagnostics and understanding modal contributions.
        """
        alpha_x = np.asarray(alpha_x, dtype=complex)
        Y_e_down, Y_h_down = self.admittance_below(alpha_x, beta)
        Y_e_up, Y_h_up = self.admittance_above(alpha_x, beta)

        F_e = 1.0 / (Y_e_down + Y_e_up)
        F_h = 1.0 / (Y_h_down + Y_h_up)
        return F_e, F_h
