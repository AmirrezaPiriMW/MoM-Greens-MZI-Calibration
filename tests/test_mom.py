"""Tests for the Galerkin MoM solver."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mzi_sensor.greens import DielectricLayer, LayeredMedium
from mzi_sensor.mom import GalerkinMoM, MoMConfig


@pytest.fixture
def air_medium():
    """Medium with air above the strip."""
    sub = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3, "FR-4")
    ch = DielectricLayer(eps_r=1.0, thickness=2.0e-3, name="air")
    return LayeredMedium(sub, ch, frequency=2.45e9, strip_width=3.0e-3)


@pytest.fixture
def water_medium():
    """Medium with water analyte."""
    sub = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3, "FR-4")
    ch = DielectricLayer.from_eps_tand(77.587, 0.106, 2.0e-3, "water")
    return LayeredMedium(sub, ch, frequency=2.45e9, strip_width=3.0e-3)


class TestMoMConfig:
    def test_defaults(self):
        cfg = MoMConfig()
        assert cfg.n_basis == 2
        assert cfg.n_quad == 500

    def test_custom(self):
        cfg = MoMConfig(n_basis=3, alpha_max=200.0)
        assert cfg.n_basis == 3
        assert cfg.alpha_max == 200.0


class TestBasisFunctions:
    def test_spectral_shape(self, air_medium):
        mom = GalerkinMoM(air_medium)
        ax = np.linspace(0.1, 1000, 100)
        B1 = mom.basis_spectral(ax, 1)
        assert B1.shape == (100,)

    def test_spectral_decay(self, air_medium):
        """Basis function should decay for large αx."""
        mom = GalerkinMoM(air_medium)
        ax_small = np.array([10.0])
        ax_large = np.array([10000.0])
        B_small = np.abs(mom.basis_spectral(ax_small, 1))
        B_large = np.abs(mom.basis_spectral(ax_large, 1))
        assert B_small > B_large


class TestQuasiStatic:
    def test_effective_permittivity_air(self, air_medium):
        """ε_eff for air superstrate should be between 1 and ε_sub."""
        mom = GalerkinMoM(air_medium)
        eps_eff = mom.effective_permittivity_quasistatic(4.4)
        assert 1.0 < np.real(eps_eff) < 4.4

    def test_initial_guess_positive(self, air_medium):
        mom = GalerkinMoM(air_medium)
        beta0 = mom.initial_guess()
        assert np.real(beta0) > 0


class TestSolver:
    def test_solve_air(self, air_medium):
        """Solver should converge for air analyte."""
        config = MoMConfig(n_basis=1, n_quad=200, alpha_max=80.0)
        mom = GalerkinMoM(air_medium, config)
        beta = mom.solve()
        # β should be real-positive, greater than k₀
        assert np.real(beta) > 0
        # ε_eff should be reasonable (between 1 and ~4)
        eps_eff = (np.real(beta) / air_medium.k0) ** 2
        assert 1.0 < eps_eff < 5.0

    def test_solve_water_higher_eps(self, water_medium):
        """Water should give higher effective permittivity than air."""
        config = MoMConfig(n_basis=1, n_quad=200, alpha_max=80.0)
        mom_water = GalerkinMoM(water_medium, config)
        beta_water = mom_water.solve()

        sub = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3)
        ch = DielectricLayer(eps_r=1.0, thickness=2.0e-3)
        air_med = LayeredMedium(sub, ch, frequency=2.45e9,
                                strip_width=3.0e-3)
        mom_air = GalerkinMoM(air_med, config)
        beta_air = mom_air.solve()

        assert np.real(beta_water) > np.real(beta_air)

    def test_propagation_constant(self, air_medium):
        """propagation_constant should return γ = α + jβ."""
        config = MoMConfig(n_basis=1, n_quad=200, alpha_max=80.0)
        mom = GalerkinMoM(air_medium, config)
        gamma = mom.propagation_constant(1.003, 0.0)
        # α ≥ 0 (attenuation)
        assert np.real(gamma) >= 0
        # β > 0 (propagation)
        assert np.imag(gamma) > 0
