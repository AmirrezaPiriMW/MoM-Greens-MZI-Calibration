"""Tests for MZI model and calibration modules."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mzi_sensor.mzi import MZIModel
from mzi_sensor.calibration import AffineCalibration
from mzi_sensor.inversion import PhysicsInversion, InversionBounds, InversionResult


class TestMZIModel:
    def test_identical_arms_cancel(self):
        """Out-of-phase MZI with identical arms should give S₂₁ ≈ 0."""
        gamma = 0.1 + 50j
        mzi = MZIModel(L_ref=0.04, L_sense=0.04, gamma_ref=gamma)
        s21 = mzi.s21(gamma)
        assert np.abs(s21) < 1e-10

    def test_different_arms(self):
        """Different γ should produce non-zero S₂₁."""
        gamma_ref = 0.1 + 50j
        gamma_sense = 0.2 + 60j
        mzi = MZIModel(L_ref=0.04, L_sense=0.04, gamma_ref=gamma_ref)
        s21 = mzi.s21(gamma_sense)
        assert np.abs(s21) > 0

    def test_db_deg_conversion(self):
        """dB/deg ↔ complex roundtrip."""
        s21 = 0.05 + 0.03j
        dB, deg = MZIModel.complex_to_db_deg(s21)
        s21_back = MZIModel.db_deg_to_complex(dB, deg)
        np.testing.assert_allclose(s21_back, s21, rtol=1e-10)

    def test_observable_shape(self):
        gamma = 0.1 + 50j
        mzi = MZIModel(L_ref=0.04, L_sense=0.04, gamma_ref=gamma)
        obs = mzi.observable(0.2 + 60j)
        assert obs.shape == (2,)


class TestAffineCalibration:
    def test_anchors_match(self):
        """Calibration should reproduce anchor points exactly."""
        m1 = np.array([0.1, 0.2])
        d1 = np.array([0.3, 0.5])
        m2 = np.array([0.5, 0.8])
        d2 = np.array([1.0, 1.5])
        cal = AffineCalibration(m1, d1, m2, d2)
        np.testing.assert_allclose(cal.apply(m1), d1, atol=1e-12)
        np.testing.assert_allclose(cal.apply(m2), d2, atol=1e-12)

    def test_inverse(self):
        """Inverse calibration should undo the forward."""
        m1 = np.array([0.1, 0.2])
        d1 = np.array([0.3, 0.5])
        m2 = np.array([0.5, 0.8])
        d2 = np.array([1.0, 1.5])
        cal = AffineCalibration(m1, d1, m2, d2)
        test = np.array([0.3, 0.4])
        np.testing.assert_allclose(
            cal.apply_inverse(cal.apply(test)), test, atol=1e-12
        )

    def test_residual_zero_at_anchor(self):
        m1 = np.array([0.1, 0.2])
        d1 = np.array([0.3, 0.5])
        m2 = np.array([0.5, 0.8])
        d2 = np.array([1.0, 1.5])
        cal = AffineCalibration(m1, d1, m2, d2)
        assert cal.residual(m1, d1) < 1e-12
        assert cal.residual(m2, d2) < 1e-12


class TestInversion:
    def test_simple_inversion(self):
        """Invert a simple linear forward model."""
        def forward(eps, tand):
            return np.array([eps * 0.01, tand * 0.5])

        inv = PhysicsInversion(forward, n_starts=10)
        target = np.array([10.0 * 0.01, 0.3 * 0.5])
        result = inv.invert(target)
        assert np.isclose(result.eps_prime, 10.0, atol=0.5)
        assert np.isclose(result.tan_delta, 0.3, atol=0.1)

    def test_inversion_result_properties(self):
        r = InversionResult(
            eps_prime=10.0, tan_delta=0.5,
            residual=1e-6, s21_model_dB=-20.0,
            s21_model_deg=45.0, converged=True,
            n_evaluations=100
        )
        assert np.isclose(r.eps_double_prime, 5.0)
        assert np.isclose(np.real(r.eps_complex), 10.0)

    def test_bounds(self):
        bounds = InversionBounds(eps_min=5.0, eps_max=50.0)
        assert bounds.scipy_bounds[0] == (5.0, 50.0)
