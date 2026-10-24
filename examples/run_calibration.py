#!/usr/bin/env python3
"""
MoM/Green's Function Calibration — Full Pipeline
=================================================

Complete calibration and inversion pipeline for the IMTL MZI
dielectric sensor at 2.45 GHz using Method of Moments with
spectral-domain Green's function.

Pipeline Steps
--------------
1. Build the multilayer microstrip model (PEC/FR-4/Strip/Channel/Air)
2. Compute propagation constants for all 9 analytes via Galerkin MoM
3. Compute S₂₁ through the out-of-phase MZI network model
4. Two-anchor affine calibration (Air + Water)
5. Physics-based multi-start inversion for validation analytes
6. Generate publication-quality plots and results table

Author: Amirreza Piri, University of Pavia
"""

import sys
import os
import time
import numpy as np

# Add parent to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mzi_sensor.greens import DielectricLayer, LayeredMedium
from mzi_sensor.mom import GalerkinMoM, MoMConfig
from mzi_sensor.mzi import MZIModel
from mzi_sensor.calibration import AffineCalibration
from mzi_sensor.inversion import PhysicsInversion, InversionBounds
from mzi_sensor.materials import (
    ANALYTES, AIR, WATER, VALIDATION_ANALYTES,
    DEFAULT_GEOMETRY, SensorGeometry,
)
from mzi_sensor.plotting import (
    plot_s21_complex_plane,
    plot_permittivity_retrieval,
    plot_error_bars,
    plot_dispersion_curve,
)


def build_medium(geometry: SensorGeometry,
                 analyte_eps: float,
                 analyte_tand: float) -> LayeredMedium:
    """Construct the LayeredMedium for a given analyte."""
    substrate = DielectricLayer.from_eps_tand(
        geometry.substrate_eps_r,
        geometry.substrate_tand,
        thickness=geometry.substrate_thickness,
        name="FR-4 substrate"
    )
    channel = DielectricLayer.from_eps_tand(
        analyte_eps, analyte_tand,
        thickness=geometry.channel_height,
        name=f"analyte(ε'={analyte_eps:.1f})"
    )
    return LayeredMedium(
        substrate=substrate,
        channel=channel,
        frequency=geometry.frequency,
        strip_width=geometry.strip_width,
    )


def main():
    """Run the full MoM/Green's Function calibration pipeline."""
    print("=" * 70)
    print("  MoM/Green's Function Calibration — IMTL MZI Sensor @ 2.45 GHz")
    print("=" * 70)

    geo = DEFAULT_GEOMETRY
    config = MoMConfig(n_basis=1, n_quad=300, alpha_max=100.0, tol=1e-10)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Step 1: Compute propagation constants for all analytes
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 1] Computing propagation constants via MoM...")
    print("─" * 55)

    gammas = []
    t_start = time.time()

    for analyte in ANALYTES:
        medium = build_medium(geo, analyte.eps_prime, analyte.tan_delta)
        mom = GalerkinMoM(medium, config)
        gamma = mom.propagation_constant(
            analyte.eps_prime, analyte.tan_delta, verbose=False
        )
        gammas.append(gamma)

        alpha_np = np.real(gamma)
        beta_rad = np.imag(gamma)
        eps_eff = (beta_rad / medium.k0) ** 2

        print(f"  {analyte.name:10s}  ε'={analyte.eps_prime:6.3f}  "
              f"β/k₀={beta_rad/medium.k0:.4f}  "
              f"α={alpha_np:.3f} Np/m  "
              f"ε_eff={eps_eff:.3f}")

    t_mom = time.time() - t_start
    print(f"\n  MoM solver time: {t_mom:.2f} s")

    # ──────────────────────────────────────────────────────────────
    # Step 2: Compute S₂₁ through MZI model
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 2] Computing MZI S₂₁ response...")
    print("─" * 55)

    # Reference arm uses Air propagation constant
    gamma_ref = gammas[0]  # Air
    mzi = MZIModel(L_ref=geo.L_ref, L_sense=geo.L_sense,
                   gamma_ref=gamma_ref)

    model_s21 = []
    model_obs = []

    for i, (analyte, gamma) in enumerate(zip(ANALYTES, gammas)):
        s21 = mzi.s21(gamma)
        s21_dB, s21_deg = MZIModel.complex_to_db_deg(s21)
        model_s21.append(s21)
        model_obs.append(mzi.observable(gamma))

        print(f"  {analyte.name:10s}  "
              f"Model: {s21_dB:8.3f} dB, {s21_deg:8.3f}°  |  "
              f"Meas:  {analyte.s21_dB:8.3f} dB, {analyte.s21_deg:8.3f}°")

    # ──────────────────────────────────────────────────────────────
    # Step 3: Two-anchor affine calibration (Air + Water)
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 3] Two-anchor affine calibration (Air + Water)...")
    print("─" * 55)

    # Observables at anchors
    obs_model_air = model_obs[0]
    obs_meas_air = np.array([np.real(AIR.s21_complex),
                              np.imag(AIR.s21_complex)])

    obs_model_water = model_obs[-1]
    obs_meas_water = np.array([np.real(WATER.s21_complex),
                                np.imag(WATER.s21_complex)])

    cal = AffineCalibration(
        obs_model_anchor1=obs_model_air,
        obs_meas_anchor1=obs_meas_air,
        obs_model_anchor2=obs_model_water,
        obs_meas_anchor2=obs_meas_water,
    )
    print(cal.summary())

    # Apply calibration to all analytes
    cal_s21 = []
    print("\n  Calibrated model S₂₁:")
    for i, (analyte, obs_m) in enumerate(zip(ANALYTES, model_obs)):
        obs_cal = cal.apply(obs_m)
        s21_cal = obs_cal[0] + 1j * obs_cal[1]
        cal_s21.append(s21_cal)
        s21_cal_dB, s21_cal_deg = MZIModel.complex_to_db_deg(s21_cal)
        residual = cal.residual(obs_m, np.array([np.real(analyte.s21_complex),
                                                  np.imag(analyte.s21_complex)]))
        print(f"  {analyte.name:10s}  "
              f"Cal: {s21_cal_dB:8.3f} dB, {s21_cal_deg:8.3f}°  "
              f"Residual: {residual:.4e}")

    # ──────────────────────────────────────────────────────────────
    # Step 4: Physics-based inversion for validation analytes
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 4] Physics-based inversion (multi-start)...")
    print("─" * 55)

    def forward_model(eps_prime, tan_delta):
        """MoM forward model: (ε', tanδ) → calibrated [Re(S₂₁), Im(S₂₁)]."""
        medium = build_medium(geo, eps_prime, tan_delta)
        mom_solver = GalerkinMoM(medium, config)
        gamma = mom_solver.propagation_constant(eps_prime, tan_delta)
        obs_raw = mzi.observable(gamma)
        return cal.apply(obs_raw)

    inverter = PhysicsInversion(
        forward_model=forward_model,
        bounds=InversionBounds(eps_min=1.0, eps_max=100.0,
                               tand_min=0.0, tand_max=2.0),
        n_starts=15,
    )

    results = []
    validation = VALIDATION_ANALYTES

    for analyte in validation:
        print(f"\n  Inverting {analyte.name}...")
        obs_meas = np.array([np.real(analyte.s21_complex),
                              np.imag(analyte.s21_complex)])
        result = inverter.invert(obs_meas, verbose=True)
        results.append(result)

        eps_err = (result.eps_prime - analyte.eps_prime) / analyte.eps_prime * 100
        td_err = result.tan_delta - analyte.tan_delta
        print(f"  → {result.summary()}")
        print(f"    ε' error: {eps_err:+.2f}%,  tan δ error: {td_err:+.4f}")

    # ──────────────────────────────────────────────────────────────
    # Step 5: Results summary table
    # ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("  RETRIEVAL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Analyte':>10s} | {'ε_ref':>7s} {'ε_ret':>7s} {'Err%':>7s} | "
          f"{'td_ref':>7s} {'td_ret':>7s} {'Δtanδ':>7s} | {'Residual':>10s}")
    print("─" * 90)

    eps_errors = []
    for analyte, result in zip(validation, results):
        eps_err = (result.eps_prime - analyte.eps_prime) / analyte.eps_prime * 100
        td_err = result.tan_delta - analyte.tan_delta
        eps_errors.append(abs(eps_err))
        print(f"{analyte.name:>10s} | {analyte.eps_prime:7.3f} "
              f"{result.eps_prime:7.3f} {eps_err:+7.2f} | "
              f"{analyte.tan_delta:7.4f} {result.tan_delta:7.4f} "
              f"{td_err:+7.4f} | {result.residual:10.2e}")

    print("─" * 90)
    print(f"  Mean |ε' error|: {np.mean(eps_errors):.2f}%")
    print(f"  Max  |ε' error|: {np.max(eps_errors):.2f}%")

    # ──────────────────────────────────────────────────────────────
    # Step 6: Generate plots
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 6] Generating plots...")

    plot_s21_complex_plane(
        ANALYTES, model_s21=model_s21, cal_s21=cal_s21,
        save_path="output/fig_s21_complex.png",
        title="S₂₁ Complex Plane — MoM/Green's Function @ 2.45 GHz"
    )
    print("  → output/fig_s21_complex.png")

    plot_permittivity_retrieval(
        validation, results,
        save_path="output/fig_retrieval.png"
    )
    print("  → output/fig_retrieval.png")

    plot_error_bars(
        validation, results,
        save_path="output/fig_errors.png"
    )
    print("  → output/fig_errors.png")

    print("\n✓ Pipeline complete.")
    return results


if __name__ == "__main__":
    main()
