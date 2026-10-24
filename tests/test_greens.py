"""Tests for the spectral-domain Green's function module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mzi_sensor.greens import (
    DielectricLayer, LayeredMedium, SpectralGreenFunction,
    MU_0, EPS_0, C_0,
)


@pytest.fixture
def medium():
    """Standard test medium: FR-4 substrate, air channel."""
    sub = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3, "FR-4")
    ch = DielectricLayer(eps_r=1.0, thickness=2.0e-3, name="air")
    return LayeredMedium(sub, ch, frequency=2.45e9, strip_width=3.0e-3)


class TestDielectricLayer:
    def test_from_eps_tand(self):
        layer = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3)
        assert np.isclose(np.real(layer.eps_r), 4.4)
        assert np.isclose(np.imag(layer.eps_r), -4.4 * 0.02)

    def test_lossless(self):
        layer = DielectricLayer.from_eps_tand(1.0, 0.0, np.inf)
        assert np.isclose(layer.eps_r, 1.0)


class TestLayeredMedium:
    def test_wavenumber_air(self, medium):
        air = DielectricLayer(eps_r=1.0, thickness=np.inf)
        k = medium.wavenumber(air)
        assert np.isclose(np.real(k), medium.k0, rtol=1e-10)

    def test_wavenumber_dielectric(self, medium):
        k = medium.wavenumber(medium.substrate)
        expected = medium.k0 * np.sqrt(medium.substrate.eps_r)
        assert np.isclose(k, expected)

    def test_k0_value(self, medium):
        expected = 2 * np.pi * 2.45e9 / C_0
        assert np.isclose(medium.k0, expected, rtol=1e-10)


class TestSpectralGreenFunction:
    def test_kernel_shape(self, medium):
        gf = SpectralGreenFunction(medium)
        ax = np.linspace(1, 1000, 50)
        beta = medium.k0 * 2.0
        G = gf.kernel(ax, beta)
        assert G.shape == (50,)

    def test_kernel_nonzero(self, medium):
        gf = SpectralGreenFunction(medium)
        ax = np.linspace(10, 500, 20)
        beta = medium.k0 * 1.5
        G = gf.kernel(ax, beta)
        assert np.all(np.abs(G) > 0)

    def test_admittance_symmetry(self, medium):
        """TM/TE admittances should be non-zero for physical αx."""
        gf = SpectralGreenFunction(medium)
        ax = np.array([100.0, 200.0, 500.0])
        beta = medium.k0 * 2.0
        Ye_d, Yh_d = gf.admittance_below(ax, beta)
        Ye_u, Yh_u = gf.admittance_above(ax, beta)
        assert np.all(np.abs(Ye_d) > 0)
        assert np.all(np.abs(Yh_d) > 0)
        assert np.all(np.abs(Ye_u) > 0)
        assert np.all(np.abs(Yh_u) > 0)

    def test_kernel_decomposed(self, medium):
        gf = SpectralGreenFunction(medium)
        ax = np.array([50.0, 100.0, 200.0])
        beta = medium.k0 * 1.8
        Fe, Fh = gf.kernel_decomposed(ax, beta)
        assert Fe.shape == (3,)
        assert Fh.shape == (3,)
        # Verify consistency with full kernel
        G = gf.kernel(ax, beta)
        G_manual = (beta**2 * Fe + ax**2 * Fh) / (ax**2 + beta**2)
        np.testing.assert_allclose(G, G_manual, rtol=1e-10)

    def test_high_permittivity_channel(self):
        """Green's function should work for high-permittivity analytes."""
        sub = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3)
        ch = DielectricLayer.from_eps_tand(80.0, 0.1, 2.0e-3)
        med = LayeredMedium(sub, ch, frequency=2.45e9, strip_width=3.0e-3)
        gf = SpectralGreenFunction(med)
        ax = np.linspace(10, 500, 20)
        beta = med.k0 * 3.0
        G = gf.kernel(ax, beta)
        assert np.all(np.isfinite(G))
