#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2
from joblib import Parallel, delayed


def main() -> None:
    """Run the Rayleigh–Taylor instability simulation and save each density
    snapshot as a .npy file in the logged_frames directory.
    """
    # User-configurable parameters
    Nx: int = 128  # Grid points in each dimension (Nx = Ny)
    Lx: float = 1.0  # Domain size in x
    Ly: float = 1.0  # Domain size in y
    g: float = 9.81  # Gravity
    rho1: float = 1.0  # Density of bottom fluid
    rho2: float = 2.0  # Density of top fluid
    nu: float = 0.0  # Kinematic viscosity
    kappa: float = 0.0  # Density diffusivity
    dt_init: float = 0.01  # Initial time step
    tmax: float = 2.0  # Total simulation time

    # Set up output directory relative to script location
    try:
        output_dir = Path(__file__).resolve().parent / "logged_frames"
    except NameError:
        output_dir = Path.cwd() / "logged_frames"

    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"{output_dir} exists and is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derived grid and spectral quantities
    Ny: int = Nx
    dx: float = Lx / Nx
    dy: float = Ly / Ny
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="ij")
    K2 = kx_mesh**2 + ky_mesh**2
    K2[0, 0] = 1.0  # avoid division by zero
    kmax = np.max(np.abs(kx))
    mask = (
        (np.abs(kx_mesh) <= 2 / 3 * kmax)
        & (np.abs(ky_mesh) <= 2 / 3 * kmax)
    )

    # Initialize fields
    rho0 = 0.5 * (rho1 + rho2)
    rho = np.where(Y > Ly / 2, rho2, rho1)
    rho += (
        0.01
        * np.cos(2 * np.pi * X / Lx)
        * np.exp(-((Y - Ly / 2) / (0.05 * Ly))**2)
    )
    omega = np.zeros_like(rho)

    # Time-stepping variables
    dt: float = dt_init
    t: float = 0.0
    frame_idx: int = 0

    print(f"Starting simulation: Nx={Nx}, dt_init={dt_init}, tmax={tmax}")

    # Main RK2 loop
    while t < tmax:
        # Spectral transforms with dealiasing
        w_hat = fft2(omega)
        w_hat[~mask] = 0
        rho_hat = fft2(rho)
        rho_hat[~mask] = 0

        # Streamfunction and velocity
        psi_hat = -w_hat / K2
        psi_hat[0, 0] = 0

        def comp_u() -> np.ndarray:
            return np.real(ifft2(1j * ky_mesh * psi_hat))

        def comp_v() -> np.ndarray:
            return np.real(ifft2(-1j * kx_mesh * psi_hat))

        u, v = Parallel(n_jobs=2)(delayed(f)() for f in (comp_u, comp_v))

        # Spatial derivatives
        rho_x = np.real(ifft2(1j * kx_mesh * rho_hat))
        rho_y = np.real(ifft2(1j * ky_mesh * rho_hat))
        lap_rho = np.real(ifft2(-K2 * rho_hat))
        omega_x = np.real(ifft2(1j * kx_mesh * w_hat))
        omega_y = np.real(ifft2(1j * ky_mesh * w_hat))
        lap_omega = np.real(ifft2(-K2 * w_hat))

        # RHS of vorticity and density
        domega = -(u * omega_x + v * omega_y) + (g / rho0) * rho_x + nu * lap_omega
        drho = -(u * rho_x + v * rho_y) + kappa * lap_rho

        # RK2 predictor
        omega1 = omega + dt * domega
        rho1 = rho + dt * drho

        # Predictor transforms
        w1_hat = fft2(omega1)
        w1_hat[~mask] = 0
        rho1_hat = fft2(rho1)
        rho1_hat[~mask] = 0

        psi1_hat = -w1_hat / K2
        psi1_hat[0, 0] = 0
        u1 = np.real(ifft2(1j * ky_mesh * psi1_hat))
        v1 = np.real(ifft2(-1j * kx_mesh * psi1_hat))

        rho1_x = np.real(ifft2(1j * kx_mesh * rho1_hat))
        rho1_y = np.real(ifft2(1j * ky_mesh * rho1_hat))
        lap_rho1 = np.real(ifft2(-K2 * rho1_hat))
        omega1_x = np.real(ifft2(1j * kx_mesh * w1_hat))
        omega1_y = np.real(ifft2(1j * ky_mesh * w1_hat))
        lap_omega1 = np.real(ifft2(-K2 * w1_hat))

        domega1 = -(u1 * omega1_x + v1 * omega1_y) + (
            g / rho0
        ) * rho1_x + nu * lap_omega1
        drho1 = -(u1 * rho1_x + v1 * rho1_y) + kappa * lap_rho1

        # RK2 corrector
        omega += 0.5 * dt * (domega + domega1)
        rho += 0.5 * dt * (drho + drho1)
        t += dt

        # CFL-based dt adjustment
        h = min(dx, dy)
        umax = np.max(np.sqrt(u**2 + v**2))
        dt_max = 0.5 * h / (umax + 1e-8)
        dt = min(dt, dt_max)

        # Save density frame
        filename = output_dir / f"frame_{frame_idx:05d}.npy"
        np.save(str(filename), rho)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"t={t:.3f}, saved {frame_idx} frames, dt={dt:.5f}")

    print(
        f"Simulation complete. Total frames: {frame_idx} in '{output_dir}'"
    )


if __name__ == "__main__":
    main()
