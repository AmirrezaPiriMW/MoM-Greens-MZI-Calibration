# Mathematical Formulation

## 1. Problem Statement

Given a measured S₂₁ from the IMTL MZI dielectric sensor, recover the complex permittivity ε = ε' − jε″ of the analyte using a physics-based forward model instead of empirical polynomial fitting.

## 2. Multilayer Structure

The microstrip cross-section consists of (bottom to top):

| Layer | Material | Parameters |
|-------|----------|------------|
| 0 | PEC ground plane | Perfect conductor |
| 1 | Substrate (FR-4) | ε_r = 4.4(1 − j·0.02), h = 1.6 mm |
| — | Strip conductor | Width W = 3.0 mm |
| 2 | Analyte channel | ε_a = ε' − jε″, h = 2.0 mm |
| 3 | Air half-space | ε_r = 1 |

## 3. Spectral-Domain Green's Function

### 3.1 Field Decomposition

Following Itoh (1980), the electromagnetic fields are decomposed into:
- **e-type (TM to y):** E_y, E_z, H_x components
- **h-type (TE to y):** H_y, H_z, E_x components

### 3.2 Transverse Propagation Constant

In each layer i with wavenumber k_i = k₀√ε_r,i:

    γ_y = √(α_x² + β² − k_i²)

with the branch chosen so that Re(γ_y) ≥ 0 (proper wave).

### 3.3 Characteristic Admittances

**TM (e-type):**
    Y_e = jωε_i / γ_y

**TE (h-type):**
    Y_h = γ_y / (jωμ₀)

### 3.4 Transmission-Line Cascade

For a layer of thickness d terminated by Y_load:

    Y_in = Y₀ · (Y_load + Y₀·tanh(γ_y·d)) / (Y₀ + Y_load·tanh(γ_y·d))

For the PEC-backed substrate (short-circuit):

    Y_in = Y₀ · coth(γ_y·d)

### 3.5 Green's Function Kernel

    G̃_zz(α_x, β) = [β²·F_e + α_x²·F_h] / (α_x² + β²)

where:
    F_e = 1 / (Y_e↓ + Y_e↑)    (TM response)
    F_h = 1 / (Y_h↓ + Y_h↑)    (TE response)

Y_e↓, Y_h↓ are the admittances looking down into the substrate + PEC ground.
Y_e↑, Y_h↑ are the admittances looking up into the analyte channel + air.

## 4. Galerkin Method of Moments

### 4.1 Surface Current Expansion

The z-directed surface current on the strip is expanded as:

    J_z(x) = Σ_n a_n · B_n(x),    |x| ≤ W/2

with entire-domain basis functions:

    B_n(x) = cos((2n−1)πx / W)

### 4.2 Galerkin Matrix

The impedance matrix elements are:

    Z_mn(β) = ∫_{-∞}^{∞} B̃_m(−α_x) · G̃_zz(α_x, β) · B̃_n(α_x) dα_x

where B̃_n(α_x) is the Fourier transform of the n-th basis function.

### 4.3 Dispersion Equation

The propagation constant β is found from:

    det[Z(β)] = 0

For a single basis function (N=1), this reduces to Z₁₁(β) = 0.

### 4.4 Root-Finding

The dispersion equation is solved using:
1. **Initial guess** from Hammerstad-Jensen quasi-static formula
2. **Root search** using scipy.optimize.fsolve (Python) or fsolve (MATLAB)

## 5. MZI Network Model

### 5.1 Out-of-Phase Topology

    S₂₁ = ½ [exp(−γ_ref · L_ref) − exp(−γ_sense · L_sense)]

where γ = α + jβ is the complex propagation constant.

### 5.2 Interferometric Null

When both arms carry identical γ and L, S₂₁ = 0 (perfect cancellation). The sensitivity arises from perturbation of the sensing arm's γ by the analyte.

## 6. Two-Anchor Affine Calibration

### 6.1 Calibration Map

    [Re(S₂₁)]_meas = a_Re · [Re(S₂₁)]_model + b_Re
    [Im(S₂₁)]_meas = a_Im · [Im(S₂₁)]_model + b_Im

### 6.2 Anchor Substances

- **Anchor 1:** Air (ε' = 1.003, tan δ = 0)
- **Anchor 2:** Water (ε' = 77.587, tan δ = 0.106)

### 6.3 Coefficient Computation

    a = (obs_meas₂ − obs_meas₁) / (obs_model₂ − obs_model₁)
    b = obs_meas₁ − a · obs_model₁

## 7. Physics-Based Inversion

### 7.1 Objective Function

    f(ε', tan δ) = ‖cal(S₂₁_model(ε', tan δ)) − S₂₁_meas‖²

### 7.2 Bounds

    1 ≤ ε' ≤ 100
    0 ≤ tan δ ≤ 2

### 7.3 Multi-Start Optimization

Latin Hypercube Sampling generates N initial points across the bounded domain, and L-BFGS-B optimization finds the global minimum.

## References

1. T. Itoh, "Spectral Domain Immittance Approach for Dispersion Characteristics of Generalized Printed Transmission Lines," IEEE Trans. MTT, vol. 28, no. 7, pp. 733-736, Jul. 1980.
2. T. Itoh and R. Mittra, "Spectral-Domain Approach for Calculating the Dispersion Characteristics of Microstrip Lines," IEEE Trans. MTT, vol. 21, no. 7, pp. 496-499, Jul. 1973.
3. D. M. Pozar, "Microwave Engineering," 4th ed., Wiley, 2012.
4. R. E. Collin, "Foundations for Microwave Engineering," 2nd ed., IEEE Press, 2001.
5. R. H. Jansen, "The Spectral-Domain Approach for Microwave Integrated Circuits," IEEE Trans. MTT, vol. 33, no. 10, 1985.
