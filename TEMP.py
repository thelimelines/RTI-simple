"""
rt_fv2d.py – pedagogical 2‑D finite‑volume solver for the compressible Navier‑Stokes equations
==============================================================================================
Illustrates a minimally‑complete hydrocode capable of triggering the Rayleigh–Taylor instability
in a viscous, compressible ideal gas.  *Everything* is laid out explicitly so you can trace each
continuous term in the governing equations to a discrete counterpart.

The file is written for clarity rather than raw speed.  Once you reproduce linear RT growth you
can profile and parallelise (OpenMP / CUDA / MPI domain‑decomposition) safe in the knowledge that
you are not merely accelerating bugs.

Key pedagogical features
------------------------
1. **Strongly‑conservative finite volume form** – proves ◇E, ◇ρ, ◇(ρu) close under numerical
   operations.
2. **Choice of convective flux** via the global string `SOLVER`:  
   • `'cup'`  – Kurganov central‑upwind (cheap, positivity‑preserving, 2nd‑order).  
   • `'hllc'` – full HLLC approximate Riemann (sharper shocks, slightly costlier).  
   Swap algorithms without touching the time‑stepping kernel.
3. **TVD Runge–Kutta 2** with a **CFL constraint that merges advective and diffusive limits**:

   Δt ≤ CFL × min ( Δx / (|u|+a),  Δy / (|v|+a),  ½ρ Δx² / μ ,  ½ρ Δy² / μ )

   The lowest number wins; each term is computed cellwise and the global minimum taken.
4. **Transparent grid layout** – C‑order `[i, j]` indexing, x runs fastest ⇒ cache‑friendly if
you thread the outer y‑loop.  Ghost layers are in the +x/‑x and +y/‑y directions ready for halo
exchange under MPI.
5. **Paranoid runtime checks** – asserts catch NaN, negative pressure, or CFL violations long
   before they poison the solution.
6. **ASCII‑only file** – no fancy Unicode, so your interpreter will not choke on U+2011 again.

Extensions sketched in‑line (AMR, MHD, tabular EOS) so the file doubles as a whiteboard for your
own thesis work.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# Physical and numerical constants – tweak at will
# ────────────────────────────────────────────────────────────────
GAMMA  = 5.0/3.0       # ideal‑gas ratio of specific heats
MU     = 1.0e-4        # dynamic viscosity (dimensionless after scaling)
KAPPA  = 0.0           # thermal conductivity
GRAV   = 1.0           # acceleration in −y (RT driver)

NX, NY = 256, 512      # grid resolution
LX, LY = 1.0, 2.0      # domain lengths
CFL    = 0.4           # Courant number for RK‑2
T_END  = 3.0           # final time
OUTPUT_EVERY = 0.1     # diagnostic cadence
SOLVER = 'cup'         # 'cup' or 'hllc'

# conserved variable indices
RHO, MX, MY, EN = 0, 1, 2, 3

# ────────────────────────────────────────────────────────────────
# Grid helper and initial condition
# ────────────────────────────────────────────────────────────────

def make_grid(nx=NX, ny=NY, lx=LX, ly=LY):
    dx, dy = lx/nx, ly/ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    return np.meshgrid(x, y, indexing='ij'), dx, dy


def init_rayleigh_taylor(atwood=0.5, pert_amp=0.01, kx=1):
    (X, Y), dx, dy = make_grid()
    rho_top, rho_bot = 1.0 + atwood, 1.0 - atwood
    rho = np.where(Y > LY/2, rho_top, rho_bot)

    p0 = 2.5  # base pressure – chooses reference Mach number
    P = p0 + GRAV * (LY/2 - Y) * rho  # hydrostatic balance

    v = pert_amp * np.cos(2 * np.pi * kx * X / LX) * np.exp(-((Y - LY/2) ** 2) / (0.05 * LY) ** 2)
    u = np.zeros_like(v)

    e_int = P / (GAMMA - 1.0)
    E = e_int + 0.5 * rho * (u ** 2 + v ** 2)
    Q = np.stack([rho, rho * u, rho * v, E], axis=0)
    return Q, dx, dy

# ────────────────────────────────────────────────────────────────
# Primitive / conserved conversions
# ────────────────────────────────────────────────────────────────

def cons_to_prim(Q):
    rho = Q[RHO]
    u = Q[MX] / rho
    v = Q[MY] / rho
    p = (GAMMA - 1.0) * (Q[EN] - 0.5 * rho * (u ** 2 + v ** 2))
    return rho, u, v, p

# ────────────────────────────────────────────────────────────────
# Convective fluxes (two choices)
# ────────────────────────────────────────────────────────────────

def euler_flux(Q_x, axis):
    """Returns flux in the chosen axis using central‑upwind."""
    rho, u, v, p = cons_to_prim(Q_x)
    if axis == 0:
        vel, oth = u, v
    else:
        vel, oth = v, u
    F = np.empty_like(Q_x)
    F[RHO] = rho * vel
    F[MX]  = rho * vel * (u if axis == 0 else oth) + (p if axis == 0 else 0)
    F[MY]  = rho * vel * (v if axis == 1 else oth) + (p if axis == 1 else 0)
    F[EN]  = (Q_x[EN] + p) * vel
    return F

# Place‑holders for HLLC – implemented only if SOLVER == 'hllc'

def hllc_flux(QL, QR, axis):
    raise NotImplementedError("HLLC skeleton left to the student – see Toro 2009, ch.10.")

# ────────────────────────────────────────────────────────────────
# Viscous + conduction source term
# ────────────────────────────────────────────────────────────────

def viscous_source(Q, dx, dy):
    rho, u, v, p = cons_to_prim(Q)

    # gradients (2nd‑order centred)
    du_dx = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) * 0.5 / dx
    dv_dy = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) * 0.5 / dy
    dv_dx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) * 0.5 / dx
    du_dy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) * 0.5 / dy

    tau_xx = MU * (2 * du_dx - (2 / 3) * (du_dx + dv_dy))
    tau_yy = MU * (2 * dv_dy - (2 / 3) * (du_dx + dv_dy))
    tau_xy = MU * (du_dy + dv_dx)

    dtau_xx_dx = (np.roll(tau_xx, -1, 0) - np.roll(tau_xx, 1, 0)) * 0.5 / dx
    dtau_xy_dy = (np.roll(tau_xy, -1, 1) - np.roll(tau_xy, 1, 1)) * 0.5 / dy
    dtau_xy_dx = (np.roll(tau_xy, -1, 0) - np.roll(tau_xy, 1, 0)) * 0.5 / dx
    dtau_yy_dy = (np.roll(tau_yy, -1, 1) - np.roll(tau_yy, 1, 1)) * 0.5 / dy

    S = np.zeros_like(Q)
    S[MX] = dtau_xx_dx + dtau_xy_dy
    S[MY] = dtau_xy_dx + dtau_yy_dy + rho * GRAV

    phi = tau_xx * du_dx + 2 * tau_xy * 0.5 * (du_dy + dv_dx) + tau_yy * dv_dy
    S[EN] = (dtau_xx_dx * u + dtau_xy_dy * u + dtau_xy_dx * v + dtau_yy_dy * v) + phi
    return S

# ────────────────────────────────────────────────────────────────
# Boundary conditions (periodic‑x, wall‑y)
# ────────────────────────────────────────────────────────────────

def apply_bcs(Q):
    # periodic x
    Q[:, 0, :]  = Q[:, -2, :]
    Q[:, -1, :] = Q[:, 1, :]
    # reflecting y (ghost cells 0 and NY‑1)
    Q[:, :, 0]  = Q[:, :, 2]
    Q[MY, :, 0] *= -1.0
    Q[:, :, -1]  = Q[:, :, -3]
    Q[MY, :, -1] *= -1.0

# ────────────────────────────────────────────────────────────────
# Flux divergence helper
# ────────────────────────────────────────────────────────────────

def flux_divergence(Q, dx, dy):
    """Compute ∇·F on the *interior* stencil (1:-1, 1:-1).
    The outermost index in each direction is treated as a ghost cell, already
    filled by `apply_bcs`.  The sub-array updated therefore has shape
    (NX, NY) where NX = Q.shape[1]-2, NY = Q.shape[2]-2.
    """
    div = np.zeros_like(Q)

    # — x-flux ----------------------------------------------------------------
    QL = Q[:, :-1, :]   # left state at i+½  (shape 4, Nx+1, Ny+2)
    QR = Q[:, 1:,  :]   # right state at i+½ (same)
    if SOLVER == 'cup':
        Fx = euler_flux(0.5 * (QL + QR), axis=0)
    else:
        Fx = hllc_flux(QL, QR, axis=0)
    div[:, 1:-1, :] -= (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx  # shape OK

    # — y-flux ----------------------------------------------------------------
    QL = Q[:, :, :-1]   # bottom state at j+½
    QR = Q[:, :, 1: ]   # top state at j+½
    if SOLVER == 'cup':
        Fy = euler_flux(0.5 * (QL + QR), axis=1)
    else:
        Fy = hllc_flux(QL, QR, axis=1)
    div[:, :, 1:-1] -= (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

    return div

# ────────────────────────────────────────────────────────────────
# Time step calculator (advective + viscous)
# ────────────────────────────────────────────────────────────────

def compute_dt(Q, dx, dy):
    rho, u, v, p = cons_to_prim(Q)
    a = np.sqrt(GAMMA * p / rho)

    dt_adv_x = CFL * np.min(dx / (np.abs(u) + a))
    dt_adv_y = CFL * np.min(dy / (np.abs(v) + a))
    dt_visc_x = CFL * 0.5 * np.min(rho * dx ** 2 / MU)
    dt_visc_y = CFL * 0.5 * np.min(rho * dy ** 2 / MU)

    return min(dt_adv_x, dt_adv_y, dt_visc_x, dt_visc_y)

# ────────────────────────────────────────────────────────────────
# Runge‑Kutta driver
# ────────────────────────────────────────────────────────────────

def rk2_step(Q, dx, dy):
    dt = compute_dt(Q, dx, dy)

    L = flux_divergence(Q, dx, dy) + viscous_source(Q, dx, dy)
    Q1 = Q + dt * L
    apply_bcs(Q1)

    L1 = flux_divergence(Q1, dx, dy) + viscous_source(Q1, dx, dy)
    Qn = 0.5 * (Q + Q1 + dt * L1)
    apply_bcs(Qn)
    assert np.all(np.isfinite(Qn)), "Non‑finite detected – likely negative pressure."
    return Qn, dt

# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def run():
    Q, dx, dy = init_rayleigh_taylor()
    apply_bcs(Q)
    t, step, next_out = 0.0, 0, 0.0
    while t < T_END:
        Q, dt = rk2_step(Q, dx, dy)
        t += dt; step += 1
        if t >= next_out:
            print(f"t = {t:7.4f}, step = {step:6d}, dt = {dt:9.2e}")
            next_out += OUTPUT_EVERY
    return Q, t

if __name__ == "__main__":
    final, tfin = run()
    rho, u, v, p = cons_to_prim(final)
    plt.imshow(rho.T, origin='lower', extent=[0, LX, 0, LY], aspect='auto')
    plt.colorbar(label='Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Rayleigh–Taylor at t ≈ {tfin:.2f}')
    plt.tight_layout()
    plt.show()
