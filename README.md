# MoM/Green's Function Calibration for IMTL MZI Dielectric Sensor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Physics-based calibration and permittivity inversion framework for the **IMTL Mach-Zehnder Interferometric (MZI) dielectric sensor** at **2.45 GHz**, using the **Method of Moments (MoM)** with **spectral-domain Green's function**.

## Overview

This package provides a complete pipeline for calibrating microstrip-based MZI dielectric sensors using rigorous electromagnetic analysis instead of empirical polynomial fitting. The approach uses:

1. **Spectral-domain Green's function** (Itoh's immittance approach) for the multilayer microstrip structure
2. **Galerkin Method of Moments** to solve the dispersion equation for the complex propagation constant
3. **Out-of-phase MZI network model** to compute S₂₁ from propagation constants
4. **Two-anchor affine calibration** (Air + Water) to align model with measurements
5. **Multi-start bounded inversion** to recover complex permittivity from measured S₂₁

### Advantages over Polynomial Surface Fitting (Path A)

| Feature | Polynomial (Path A) | MoM/Green's (This Work) |
|---------|-------------------|-------------------------|
| Basis | Empirical polynomial surface | Maxwell's equations |
| Extrapolation | Poor beyond measured locus | Reliable (physics-based) |
| New geometries | Requires re-measurement | Re-run with new dimensions |
| Physical insight | None | Full EM field analysis |

## Structure

```
MoM-Greens-MZI-Calibration/
├── mzi_sensor/
│   ├── __init__.py          # Package exports
│   ├── greens.py            # Spectral-domain Green's function
│   ├── mom.py               # Galerkin MoM dispersion solver
│   ├── mzi.py               # MZI S-parameter network model
│   ├── calibration.py       # Two-anchor affine calibration
│   ├── inversion.py         # Multi-start bounded inversion
│   ├── materials.py         # Material database (9 analytes)
│   └── plotting.py          # Publication-quality plots
├── examples/
│   └── run_calibration.py   # Full pipeline example
├── tests/
│   ├── test_greens.py       # Green's function tests
│   ├── test_mom.py          # MoM solver tests
│   └── test_mzi.py          # MZI, calibration, inversion tests
├── scripts/
│   └── generate_figures.py  # Regenerate all output figures
├── output/
│   ├── fig_s21_complex.png  # S₂₁ complex-plane plot
│   ├── fig_s21_comparison.png # Magnitude/phase bar charts
│   ├── fig_retrieval.png    # Permittivity retrieval scatter
│   ├── fig_errors.png       # Retrieval error bars
│   ├── fig_dispersion.png   # Dispersion curve
│   ├── fig_greens_kernel.png # Green's function kernel
│   └── results.json         # All computed data (JSON)
├── matlab/
│   └── MoM_Greens_Calibration.m  # MATLAB translation
└── docs/
    └── theory.md            # Mathematical formulation
```

## Installation

```bash
git clone https://github.com/amirrezapiri/MoM-Greens-MZI-Calibration.git
cd MoM-Greens-MZI-Calibration
pip install -e .
```

### Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

## Quick Start

```python
from mzi_sensor import (
    SpectralGreenFunction, LayeredMedium, DielectricLayer,
    GalerkinMoM, MoMConfig, MZIModel,
    AffineCalibration, PhysicsInversion,
)

# Define the multilayer structure
substrate = DielectricLayer.from_eps_tand(4.4, 0.02, 1.6e-3, "FR-4")

# Solve for air reference arm
channel_air = DielectricLayer.from_eps_tand(1.003, 0.0, 2.0e-3, "air")
medium_air = LayeredMedium(substrate, channel_air, frequency=2.45e9, strip_width=3e-3)
mom_air = GalerkinMoM(medium_air, MoMConfig(n_basis=1))
gamma_air = mom_air.propagation_constant(eps_prime=1.003, tan_delta=0.0)

# Solve for analyte sensing arm
channel = DielectricLayer.from_eps_tand(20.0, 0.5, 2.0e-3, "analyte")
medium = LayeredMedium(substrate, channel, frequency=2.45e9, strip_width=3e-3)
mom = GalerkinMoM(medium, MoMConfig(n_basis=1))
gamma = mom.propagation_constant(eps_prime=20.0, tan_delta=0.5)

# Compute MZI response
mzi = MZIModel(L_ref=40e-3, L_sense=40e-3, gamma_ref=gamma_air)
s21 = mzi.s21(gamma)
```

See [`examples/run_calibration.py`](examples/run_calibration.py) for the complete pipeline.

## Theory

### Multilayer Structure

```
                    Air half-space (ε = 1)
    ─────────────────────────────────────────
    │         Analyte channel (ε_a, h_ch)   │
    ═════════════════════════════════════════
    │    Strip conductor (width W)          │
    ─────────────────────────────────────────
    │         Substrate (ε_sub, h_sub)      │
    ═════════════════════════════════════════
                PEC ground plane
```

### Spectral-Domain Green's Function

The Green's function kernel in the spectral domain:

$$\tilde{G}_{zz}(\alpha_x, \beta) = \frac{\beta^2 F_e + \alpha_x^2 F_h}{\alpha_x^2 + \beta^2}$$

where $F_e$ and $F_h$ are the TM and TE immittance response functions obtained by cascading layer admittances via the transmission-line analogy.

### Dispersion Equation

The Galerkin MoM formulation leads to:

$$Z_{mn}(\beta) = \int_{-\infty}^{\infty} \tilde{B}_m(-\alpha_x) \cdot \tilde{G}_{zz}(\alpha_x, \beta) \cdot \tilde{B}_n(\alpha_x) \, d\alpha_x$$

The propagation constant is found from $\det[\mathbf{Z}(\beta)] = 0$.

### MZI Model

Out-of-phase topology:

$$S_{21} = \frac{1}{2} \left[ e^{-\gamma_{\text{ref}} L_{\text{ref}}} - e^{-\gamma_{\text{sense}} L_{\text{sense}}} \right]$$

## Calibration Data

9 analytes at 2.45 GHz, 29 ± 1 °C:

| Analyte | ε' | tan δ | S₂₁ [dB] | S₂₁ [°] |
|---------|-----|-------|----------|---------|
| Air | 1.003 | 0.000 | −7.605 | −54.380 |
| IPA | 5.294 | 0.757 | −16.890 | −33.390 |
| 80% IPA | 16.680 | 0.590 | −27.186 | −5.310 |
| 90% EtOH | 17.844 | 0.680 | −28.216 | 13.559 |
| Acetone | 21.925 | 0.034 | −39.647 | −27.719 |
| 60% IPA | 31.999 | 0.396 | −31.622 | 40.924 |
| 40% IPA | 48.406 | 0.285 | −30.121 | 66.664 |
| 20% IPA | 64.667 | 0.189 | −30.226 | 73.713 |
| Water | 77.587 | 0.106 | −29.293 | 78.049 |

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Author

**Amirreza Piri**  
University of Pavia  
amirreza.piri01@universitadipavia.it

## License

MIT License — see [LICENSE](LICENSE) for details.

## References

1. T. Itoh, "Spectral Domain Immittance Approach for Dispersion Characteristics of Generalized Printed Transmission Lines," IEEE Trans. MTT, vol. 28, no. 7, 1980.
2. T. Itoh and R. Mittra, "Spectral-Domain Approach for Calculating the Dispersion Characteristics of Microstrip Lines," IEEE Trans. MTT, vol. 21, no. 7, 1973.
3. D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012.
