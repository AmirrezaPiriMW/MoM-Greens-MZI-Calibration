"""
Physics-Based Multi-Start Bounded Inversion Engine
===================================================

Given a measured S₂₁ (calibrated), this module recovers the complex
permittivity (ε', tan δ) of an unknown analyte by inverting the
MoM forward model.

The inversion minimises:

    f(ε', tan δ) = ‖ cal(S₂₁_model(ε', tan δ)) − S₂₁_meas ‖²

subject to physical bounds:

    1 ≤ ε' ≤ 100,    0 ≤ tan δ ≤ 2

Multi-start optimisation with Latin Hypercube Sampling (LHS) initial
points ensures the global minimum is found despite the nonlinear,
potentially multi-modal objective landscape.

The key advantage over polynomial surface fitting (Path A) is that
this inversion uses the physics-based MoM forward model, providing
reliable extrapolation beyond the measured calibration locus.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List


@dataclass
class InversionBounds:
    """Physical bounds for the inversion variables."""
    eps_min: float = 1.0
    eps_max: float = 100.0
    tand_min: float = 0.0
    tand_max: float = 2.0

    @property
    def scipy_bounds(self):
        """Return bounds in scipy.optimize format."""
        return [(self.eps_min, self.eps_max),
                (self.tand_min, self.tand_max)]


@dataclass
class InversionResult:
    """Result of a single inversion."""
    eps_prime: float
    tan_delta: float
    residual: float
    s21_model_dB: float
    s21_model_deg: float
    converged: bool
    n_evaluations: int

    @property
    def eps_double_prime(self) -> float:
        """Imaginary part of permittivity: ε″ = ε'·tan δ."""
        return self.eps_prime * self.tan_delta

    @property
    def eps_complex(self) -> complex:
        """Complex permittivity ε = ε' − jε″."""
        return self.eps_prime * (1.0 - 1j * self.tan_delta)

    def summary(self) -> str:
        return (f"ε' = {self.eps_prime:.3f}, tan δ = {self.tan_delta:.4f}, "
                f"ε″ = {self.eps_double_prime:.3f}, "
                f"residual = {self.residual:.2e}")


class PhysicsInversion:
    """Multi-start bounded inversion using the MoM forward model.

    Parameters
    ----------
    forward_model : callable
        Function f(eps_prime, tan_delta) → ndarray[2] returning
        the calibrated model observable [Re(S₂₁), Im(S₂₁)].
    bounds : InversionBounds, optional
        Physical bounds on (ε', tan δ).
    n_starts : int
        Number of multi-start initial points.
    """

    def __init__(self, forward_model: Callable,
                 bounds: Optional[InversionBounds] = None,
                 n_starts: int = 20):
        self.forward = forward_model
        self.bounds = bounds or InversionBounds()
        self.n_starts = n_starts

    def _objective(self, x: np.ndarray,
                   obs_meas: np.ndarray) -> float:
        """Objective function: squared Euclidean distance."""
        eps_prime, tan_delta = x
        try:
            obs_model = self.forward(eps_prime, tan_delta)
            return float(np.sum((obs_model - obs_meas)**2))
        except Exception:
            return 1e20  # penalty for failed forward model

    def _latin_hypercube(self, n: int,
                         seed: int = 42) -> np.ndarray:
        """Generate Latin Hypercube Sample initial points.

        Parameters
        ----------
        n : int
            Number of sample points.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        points : ndarray, shape (n, 2)
            Sample points in [eps_min, eps_max] × [tand_min, tand_max].
        """
        rng = np.random.RandomState(seed)
        b = self.bounds

        # LHS: divide each dimension into n equal intervals
        eps_intervals = np.linspace(b.eps_min, b.eps_max, n + 1)
        tand_intervals = np.linspace(b.tand_min, b.tand_max, n + 1)

        eps_points = np.array([
            rng.uniform(eps_intervals[i], eps_intervals[i + 1])
            for i in range(n)
        ])
        tand_points = np.array([
            rng.uniform(tand_intervals[i], tand_intervals[i + 1])
            for i in range(n)
        ])

        # Shuffle one dimension
        rng.shuffle(tand_points)

        return np.column_stack([eps_points, tand_points])

    def invert(self, obs_meas: np.ndarray,
               verbose: bool = False) -> InversionResult:
        """Run multi-start bounded inversion.

        Parameters
        ----------
        obs_meas : ndarray, shape (2,)
            Measured observable [Re(S₂₁), Im(S₂₁)].
        verbose : bool
            Print convergence information.

        Returns
        -------
        result : InversionResult
            Best inversion result across all starts.
        """
        initial_points = self._latin_hypercube(self.n_starts)
        best_result = None
        best_cost = np.inf

        scipy_bounds = self.bounds.scipy_bounds

        for i, x0 in enumerate(initial_points):
            try:
                res = optimize.minimize(
                    self._objective, x0, args=(obs_meas,),
                    method='L-BFGS-B',
                    bounds=scipy_bounds,
                    options={'maxiter': 200, 'ftol': 1e-14,
                             'gtol': 1e-10}
                )

                if res.fun < best_cost:
                    best_cost = res.fun
                    best_result = res

                    if verbose:
                        print(f"  Start {i+1}/{self.n_starts}: "
                              f"ε'={res.x[0]:.3f}, "
                              f"tanδ={res.x[1]:.4f}, "
                              f"cost={res.fun:.2e}")

            except Exception as e:
                if verbose:
                    print(f"  Start {i+1}/{self.n_starts}: failed ({e})")
                continue

        if best_result is None:
            raise RuntimeError("All inversion starts failed")

        eps_prime, tan_delta = best_result.x

        # Compute model S₂₁ at the solution for reporting
        obs_sol = self.forward(eps_prime, tan_delta)
        s21_complex = obs_sol[0] + 1j * obs_sol[1]
        s21_dB = 20.0 * np.log10(np.abs(s21_complex) + 1e-30)
        s21_deg = np.degrees(np.angle(s21_complex))

        return InversionResult(
            eps_prime=float(eps_prime),
            tan_delta=float(tan_delta),
            residual=float(np.sqrt(best_cost)),
            s21_model_dB=float(s21_dB),
            s21_model_deg=float(s21_deg),
            converged=best_result.success,
            n_evaluations=int(best_result.nfev)
        )

    def invert_from_db_deg(self, s21_dB: float, s21_deg: float,
                           verbose: bool = False) -> InversionResult:
        """Convenience: invert from dB/degrees format.

        Parameters
        ----------
        s21_dB : float
            |S₂₁| in decibels.
        s21_deg : float
            ∠S₂₁ in degrees.
        """
        mag = 10.0 ** (s21_dB / 20.0)
        phase = np.radians(s21_deg)
        s21 = mag * np.exp(1j * phase)
        obs_meas = np.array([np.real(s21), np.imag(s21)])
        return self.invert(obs_meas, verbose=verbose)
