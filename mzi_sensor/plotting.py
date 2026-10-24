"""
Visualization Utilities for MZI Sensor Calibration
===================================================

Publication-quality plots for the MoM/Green's Function calibration
framework. All plots are saved as high-resolution PNG files.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional, Tuple

from .materials import Analyte, ANALYTES
from .inversion import InversionResult


# ─── Style configuration ──────────────────────────────────────────
STYLE = {
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
}

COLORS = {
    'measured': '#2166AC',
    'model': '#B2182B',
    'calibrated': '#1B7837',
    'anchor': '#E66101',
    'error': '#D6604D',
    'residual': '#762A83',
}

MARKERS = {
    'measured': 'o',
    'model': 's',
    'calibrated': '^',
    'anchor': 'D',
}


def apply_style():
    """Apply publication-quality style settings."""
    plt.rcParams.update(STYLE)


def plot_s21_complex_plane(analytes: List[Analyte],
                           model_s21: Optional[List[complex]] = None,
                           cal_s21: Optional[List[complex]] = None,
                           save_path: str = "fig_s21_complex.png",
                           title: str = "S₂₁ Complex Plane"):
    """Plot S₂₁ in the complex plane (Re vs Im).

    Parameters
    ----------
    analytes : list of Analyte
        Measured data points.
    model_s21 : list of complex, optional
        Raw MoM model S₂₁ values.
    cal_s21 : list of complex, optional
        Calibrated model S₂₁ values.
    save_path : str
        Output file path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 8))

    # Measured
    meas_re = [np.real(a.s21_complex) for a in analytes]
    meas_im = [np.imag(a.s21_complex) for a in analytes]
    ax.scatter(meas_re, meas_im, c=COLORS['measured'],
               marker=MARKERS['measured'], s=80, zorder=5,
               label='Measured', edgecolors='white', linewidth=0.5)

    for a, x, y in zip(analytes, meas_re, meas_im):
        ax.annotate(a.name, (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color=COLORS['measured'])

    # Raw model
    if model_s21 is not None:
        mod_re = [np.real(s) for s in model_s21]
        mod_im = [np.imag(s) for s in model_s21]
        ax.scatter(mod_re, mod_im, c=COLORS['model'],
                   marker=MARKERS['model'], s=60, zorder=4,
                   label='MoM Model (raw)', edgecolors='white',
                   linewidth=0.5)

    # Calibrated model
    if cal_s21 is not None:
        cal_re = [np.real(s) for s in cal_s21]
        cal_im = [np.imag(s) for s in cal_s21]
        ax.scatter(cal_re, cal_im, c=COLORS['calibrated'],
                   marker=MARKERS['calibrated'], s=70, zorder=6,
                   label='MoM Model (calibrated)', edgecolors='white',
                   linewidth=0.5)

    ax.set_xlabel('Re(S₂₁)', fontsize=12)
    ax.set_ylabel('Im(S₂₁)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_s21_polar(analytes: List[Analyte],
                   cal_s21: Optional[List[complex]] = None,
                   save_path: str = "fig_s21_polar.png"):
    """Polar plot of S₂₁ (magnitude vs phase)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'polar': True})

    # Measured
    for a in analytes:
        theta = np.radians(a.s21_deg)
        r = 10.0 ** (a.s21_dB / 20.0)
        ax.scatter(theta, r, c=COLORS['measured'], s=60, zorder=5)
        ax.annotate(a.name, (theta, r), textcoords="offset points",
                    xytext=(8, 3), fontsize=7)

    if cal_s21 is not None:
        for s in cal_s21:
            theta = np.angle(s)
            r = np.abs(s)
            ax.scatter(theta, r, c=COLORS['calibrated'],
                       marker='^', s=50, zorder=4)

    ax.set_title('S₂₁ Polar Plot', fontsize=14, fontweight='bold',
                 pad=20)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_permittivity_retrieval(analytes: List[Analyte],
                                results: List[InversionResult],
                                save_path: str = "fig_retrieval.png"):
    """Plot retrieved vs reference permittivity.

    Creates a 2×1 subplot: ε' retrieval and tan δ retrieval.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    eps_ref = [a.eps_prime for a in analytes]
    eps_ret = [r.eps_prime for r in results]
    td_ref = [a.tan_delta for a in analytes]
    td_ret = [r.tan_delta for r in results]

    # ── ε' retrieval ──
    ax1.scatter(eps_ref, eps_ret, c=COLORS['calibrated'], s=80,
                zorder=5, edgecolors='white', linewidth=0.5)
    lims = [min(min(eps_ref), min(eps_ret)) * 0.9,
            max(max(eps_ref), max(eps_ret)) * 1.1]
    ax1.plot(lims, lims, 'k--', linewidth=1, alpha=0.5,
             label='Ideal (y = x)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Reference ε'", fontsize=12)
    ax1.set_ylabel("Retrieved ε'", fontsize=12)
    ax1.set_title("Permittivity (ε') Retrieval", fontsize=13,
                   fontweight='bold')
    ax1.legend(fontsize=10)

    for a, xr, yr in zip(analytes, eps_ref, eps_ret):
        ax1.annotate(a.name, (xr, yr), textcoords="offset points",
                     xytext=(6, 4), fontsize=7)

    # ── tan δ retrieval ──
    ax2.scatter(td_ref, td_ret, c=COLORS['model'], s=80,
                zorder=5, edgecolors='white', linewidth=0.5)
    lims_td = [min(min(td_ref), min(td_ret)) - 0.05,
               max(max(td_ref), max(td_ret)) + 0.05]
    ax2.plot(lims_td, lims_td, 'k--', linewidth=1, alpha=0.5,
             label='Ideal (y = x)')
    ax2.set_xlim(lims_td)
    ax2.set_ylim(lims_td)
    ax2.set_xlabel("Reference tan δ", fontsize=12)
    ax2.set_ylabel("Retrieved tan δ", fontsize=12)
    ax2.set_title("Loss Tangent (tan δ) Retrieval", fontsize=13,
                   fontweight='bold')
    ax2.legend(fontsize=10)

    for a, xr, yr in zip(analytes, td_ref, td_ret):
        ax2.annotate(a.name, (xr, yr), textcoords="offset points",
                     xytext=(6, 4), fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_error_bars(analytes: List[Analyte],
                    results: List[InversionResult],
                    save_path: str = "fig_errors.png"):
    """Bar chart of retrieval errors."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    names = [a.name for a in analytes]
    eps_err = [(r.eps_prime - a.eps_prime) / a.eps_prime * 100
               if a.eps_prime > 0.01 else 0
               for a, r in zip(analytes, results)]
    td_err = [(r.tan_delta - a.tan_delta)
              for a, r in zip(analytes, results)]

    x = np.arange(len(names))

    # ε' relative error
    bars1 = ax1.bar(x, eps_err, color=COLORS['error'], alpha=0.8,
                    edgecolor='white')
    ax1.set_ylabel("ε' Relative Error [%]", fontsize=12)
    ax1.set_title("Permittivity Retrieval Error", fontsize=13,
                   fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right')
    ax1.axhline(0, color='black', linewidth=0.8)

    # tan δ absolute error
    bars2 = ax2.bar(x, td_err, color=COLORS['residual'], alpha=0.8,
                    edgecolor='white')
    ax2.set_ylabel("tan δ Absolute Error", fontsize=12)
    ax2.set_title("Loss Tangent Retrieval Error", fontsize=13,
                   fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right')
    ax2.axhline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_dispersion_curve(eps_range: np.ndarray,
                          beta_over_k0: np.ndarray,
                          alpha_np_per_m: np.ndarray,
                          save_path: str = "fig_dispersion.png"):
    """Plot effective permittivity and attenuation vs analyte ε'."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(eps_range, np.real(beta_over_k0)**2, 'b-', linewidth=2)
    ax1.set_xlabel("Analyte ε'", fontsize=12)
    ax1.set_ylabel("ε_eff (effective permittivity)", fontsize=12)
    ax1.set_title("Dispersion: ε_eff vs Analyte ε'", fontsize=13,
                   fontweight='bold')

    ax2.plot(eps_range, alpha_np_per_m, 'r-', linewidth=2)
    ax2.set_xlabel("Analyte ε'", fontsize=12)
    ax2.set_ylabel("α [Np/m]", fontsize=12)
    ax2.set_title("Attenuation vs Analyte ε'", fontsize=13,
                   fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
