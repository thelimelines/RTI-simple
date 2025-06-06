#!/usr/bin/env python3
"""
Simulation of the Rayleigh-Taylor instability using a pseudospectral method with optional GIF animation.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import imageio

# User-configurable flags (for in-code configuration)
ANIMATE = True  # Set to True to generate GIF animation, False to save raw .npy frames
ANIM_DT = 0.01  # Time interval between frames in animation
FRAME_DIR = "logged_frames"  # Directory for output frames, logs, and animation
GIF_NAME = "animation.gif"  # Filename for the output GIF


def save_animation(frames: list[np.ndarray], filename: Path, fps: int = 10) -> None:
    """
    Save density frames as an animated GIF.

    Args:
        frames (list of np.ndarray): List of 2D density fields to animate.
        filename (Path): Path to output GIF file.
        fps (int): Frames per second for the GIF.
    """
    images = []
    for frame in frames:
        # Normalize to [0, 1]
        norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
        # Convert to RGBA image
        img = (plt.cm.viridis(norm) * 255).astype(np.uint8)
        images.append(img)
    imageio.mimsave(str(filename), images, format="GIF", fps=fps)


def main() -> None:
    """
    Run the Rayleigh-Taylor instability simulation.

    This function supports optional GIF animation to reduce disk I/O by
    capturing frames at specified intervals and assembling them upon
    completion. It also logs time-step adaptations to a file.

    Limitations:
    - Periodic boundary conditions only.
    - Spectral 2/3-rule dealiasing.
    - Requires sufficient memory for in-memory animation frames.

    Returns:
        None
    """
    # User-configurable parameters
    Nx: int = 128  # Grid points in each dimension
    Lx: float = 1.0  # Domain size in x
    Ly: float = 1.0  # Domain size in y
    g: float = 9.81  # Gravity acceleration
    rho1: float = 1.0  # Density of bottom fluid
    rho2: float = 2.0  # Density of top fluid
    nu: float = 0.0  # Kinematic viscosity
    kappa: float = 0.0  # Density diffusivity
    dt_init: float = 0.005  # Initial time step
    tmax: float = 2.0  # Total simulation time

    animate = ANIMATE
    anim_dt = ANIM_DT

    # Set up output directory
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    output_dir = base_dir / FRAME_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file and console
    log_file = output_dir / "simulation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger()

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
    K2[0, 0] = 1.0  # avoid division by zero in Poisson solve

    # 2/3-rule dealiasing mask
    kmax = np.max(np.abs(kx))
    mask = (np.abs(kx_mesh) <= 2 / 3 * kmax) & (np.abs(ky_mesh) <= 2 / 3 * kmax)

    # Initialize fields
    rho0 = 0.5 * (rho1 + rho2)
    rho = np.where(Y > Ly / 2, rho2, rho1)
    rho += (
        0.01 * np.cos(2 * np.pi * X / Lx) * np.exp(-(((Y - Ly / 2) / (0.05 * Ly)) ** 2))
    )
    omega = np.zeros_like(rho)

    # Time-stepping variables
    dt: float = dt_init
    t: float = 0.0
    frame_idx: int = 0

    logger.info(
        f"Starting simulation: Nx={Nx}, dt_init={dt_init}, tmax={tmax} (animate={animate})"
    )

    # Animation buffer
    anim_frames: list[np.ndarray] = []
    next_anim_t = anim_dt

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
            """
            Compute x-velocity from streamfunction in spectral space.

            Returns:
                np.ndarray: Real-space x-velocity component u.
            """
            return np.real(ifft2(1j * ky_mesh * psi_hat))

        def comp_v() -> np.ndarray:
            """
            Compute y-velocity from streamfunction in spectral space.

            Returns:
                np.ndarray: Real-space y-velocity component v.
            """
            return np.real(ifft2(-1j * kx_mesh * psi_hat))

        u, v = Parallel(n_jobs=2)(delayed(f)() for f in (comp_u, comp_v))

        # Spatial derivatives
        rho_x = np.real(ifft2(1j * kx_mesh * rho_hat))
        rho_y = np.real(ifft2(1j * ky_mesh * rho_hat))
        lap_rho = np.real(ifft2(-K2 * rho_hat))
        omega_x = np.real(ifft2(1j * kx_mesh * w_hat))
        omega_y = np.real(ifft2(1j * ky_mesh * w_hat))
        lap_omega = np.real(ifft2(-K2 * w_hat))

        # RHS of vorticity and density equations
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

        domega1 = (
            -(u1 * omega1_x + v1 * omega1_y) + (g / rho0) * rho1_x + nu * lap_omega1
        )
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
        logger.info(f"t={t:.3f}, dt adjusted to {dt:.5f}")

        # Save or buffer density frame
        if animate:
            if t >= next_anim_t:
                anim_frames.append(rho.copy())
                next_anim_t += anim_dt
        else:
            frame_file = output_dir / f"frame_{frame_idx:05d}.npy"
            np.save(str(frame_file), rho)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Saved {frame_idx} frames at t={t:.3f}")

    # Finalize animation if requested
    if animate:
        gif_path = output_dir / GIF_NAME
        save_animation(anim_frames, gif_path)
        logger.info(f"Animation saved to {gif_path}")

    logger.info("Simulation complete.")


if __name__ == "__main__":
    main()
