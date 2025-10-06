#!/usr/bin/env python3
"""
qnm_shooting_fast.py

Minimal working version for article:
- Computes |A_in(ω)| map over complex ω-plane
- Finds grid minimum candidate QNM
- Outputs qnm_ucft.png, qnm_ucft.csv, qnm_physical.csv

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import csv, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- Metric ----------------
def f_coh(r, eps, lam, gamma, M):
    return 1 - (2*M*r)/(r**2 + eps**2) + (lam/6.0)*r**2*(1 - np.exp(-gamma*r))

def m_eff(r, eps, lam, gamma, M):
    return (M*r**2)/(r**2 + eps**2) - (lam/12.0)*r**3*(1 - np.exp(-gamma*r))

def V_eff(r, ell, eps, lam, gamma, M):
    f = f_coh(r, eps, lam, gamma, M)
    meff = m_eff(r, eps, lam, gamma, M)
    if r <= 0:
        return 0
    return f * (ell*(ell+1)/r**2 - 6*meff/r**3)

# ---------------- Horizon ----------------
def find_horizon(eps, lam, gamma, M, rmax=200):
    rs = np.linspace(1e-6, rmax, 20000)
    fs = f_coh(rs, eps, lam, gamma, M)
    sign_change = np.where(np.diff(np.sign(fs)))[0]
    if len(sign_change) == 0:
        i_min = np.argmin(np.abs(fs))
        r_h = float(rs[i_min])
        logging.warning(f"No sign change; using min|f| at r={r_h:.6f}")
        return r_h
    i = sign_change[0]
    r1, r2 = rs[i], rs[i+1]
    return brentq(lambda r: f_coh(r, eps, lam, gamma, M), r1, r2)

# ---------------- Wave equation ----------------
def radial_rhs(r, y, omega, ell, eps, lam, gamma, M):
    psi, dpsi = y
    f = f_coh(r, eps, lam, gamma, M)
    V = V_eff(r, ell, eps, lam, gamma, M)
    df_dr = (f_coh(r+1e-5, eps, lam, gamma, M) - f_coh(r-1e-5, eps, lam, gamma, M)) / (2e-5)
    denom = f*f if abs(f) > 1e-15 else 1e-15
    d2psi = (-f*df_dr*dpsi - (omega**2 - V)*psi) / denom
    return [dpsi, d2psi]

def integrate_outward(r_h, omega, eps, lam, gamma, M, ell, rmax, delta):
    r0 = r_h + delta
    f1 = (f_coh(r0+1e-6, eps, lam, gamma, M) - f_coh(r0-1e-6, eps, lam, gamma, M)) / (2e-6)
    psi0 = delta**(-1j*omega/f1)
    dpsi0 = (-1j*omega/f1)*delta**(-1j*omega/f1 - 1)
    sol = solve_ivp(
        lambda r, y: radial_rhs(r, y, omega, ell, eps, lam, gamma, M),
        (r0, rmax),
        [psi0, dpsi0],
        method="DOP853",
        rtol=1e-9, atol=1e-9, max_step=0.5
    )
    return sol.t, sol.y[0, :]

# ---------------- Fit asymptotics ----------------
def fit_asymptotics(r, psi, omega, Ntail=60):
    N = min(Ntail, len(r))
    r_tail = r[-N:]
    psi_tail = psi[-N:]
    E_plus = np.exp(1j*omega*r_tail)
    E_minus = np.exp(-1j*omega*r_tail)
    A = np.vstack([E_plus, E_minus]).T
    coeffs, *_ = np.linalg.lstsq(A, psi_tail, rcond=None)
    Aout, Ain = coeffs
    return Aout, Ain

def ain_magnitude(omega, eps, lam, gamma, ell, M, r_h, rmax, delta):
    try:
        r, psi = integrate_outward(r_h, omega, eps, lam, gamma, M, ell, rmax, delta)
        _, Ain = fit_asymptotics(r, psi, omega)
        return abs(Ain)
    except Exception as e:
        logging.debug(f"A_in error ω={omega}: {e}")
        return np.inf

# ---------------- Grid scan ----------------
def grid_scan(params):
    re_vals = np.linspace(*params["grid_re_range"], params["grid_n_re"])
    im_vals = np.linspace(*params["grid_im_range"], params["grid_n_im"])
    grid = np.full((len(im_vals), len(re_vals)), np.nan)
    r_h = find_horizon(params["eps"], params["lam"], params["gamma"], params["M"])
    logging.info(f"Horizon r_h = {r_h:.6f}")
    total = len(re_vals)*len(im_vals)
    count = 0
    for i, im in enumerate(im_vals):
        for j, re in enumerate(re_vals):
            omega = complex(re, -im)
            grid[i, j] = ain_magnitude(omega, params["eps"], params["lam"], params["gamma"],
                                       params["ell"], params["M"], r_h,
                                       params["rmax"], params["delta"])
            count += 1
        logging.info(f"Row {i+1}/{len(im_vals)} complete ({100*count/total:.1f}%)")
    return re_vals, im_vals, grid, r_h

# ---------------- Main ----------------
def main():
    params = {
        "eps": 0.2, "lam": 5e-4, "gamma": 10.0,
        "ell": 2, "M": 1.0,
        "delta": 1e-3, "rmax": 200.0,  # быстрые тестовые параметры
        "grid_n_re": 15, "grid_n_im": 15,
        "grid_re_range": (0.18, 0.30),
        "grid_im_range": (0.005, 0.05),
        "tail_points": 60,
    }

    re_vals, im_vals, grid, r_h = grid_scan(params)
    log_grid = np.log10(np.clip(grid, 1e-20, 1e20))
    plt.figure(figsize=(7,5))
    plt.pcolormesh(re_vals, im_vals, log_grid, shading="auto")
    plt.xlabel("Re(ω)")
    plt.ylabel("Im(ω)")
    plt.title("log10 |A_in(ω)| for U-CFT metric")
    plt.colorbar(label="log10 |A_in|")
    plt.tight_layout()
    plt.savefig("qnm_ucft.png", dpi=150)
    logging.info("Saved qnm_ucft.png")

    idx_flat = np.nanargmin(grid)
    idx = np.unravel_index(idx_flat, grid.shape)
    best_val = grid[idx]
    best_re = re_vals[idx[1]]
    best_im = im_vals[idx[0]]
    initial_omega = complex(best_re, -best_im)
    logging.info(f"Best grid candidate: ω ≈ {initial_omega}, |A_in|={best_val:.3e}")

    with open("qnm_ucft.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps","lam","gamma","ell","M","omega_re","omega_im"])
        w.writerow([params["eps"],params["lam"],params["gamma"],params["ell"],params["M"],
                    initial_omega.real, initial_omega.imag])

    # convert to Hz for M = 10, 30, 60 M_sun
    M_sun_sec = 4.9254909e-6  # s
    with open("qnm_physical.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["M_solar","f_real_Hz","f_imag_Hz"])
        for Ms in [10, 30, 60]:
            f_re = initial_omega.real / (2*np.pi*M_sun_sec*Ms)
            f_im = initial_omega.imag / (2*np.pi*M_sun_sec*Ms)
            w.writerow([Ms, f_re, f_im])
    logging.info("Saved qnm_physical.csv")

if __name__ == "__main__":
    main()
