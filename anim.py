import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import re

# Simulation parameters for subtitle (manually copied or loaded from metadata)
sim_params = {
    "Nx": 128,
    "Lx": 1.0,
    "Ly": 1.0,
    "g": 9.81,
    "rho1": 1.0,
    "rho2": 2.0,
    "nu": 0.0,
    "kappa": 0.0,
    "dt_init": 0.005,
    "tmax": 2.0,
}

def load_frames(folder):
    folder = Path(folder)
    frame_files = sorted(folder.glob("frame_*.npy"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
    return [np.load(str(f)) for f in frame_files]

def main():
    frames = load_frames("logged_frames")
    if not frames:
        raise FileNotFoundError("No .npy frames found in 'logged_frames'.")

    Nx = frames[0].shape[0]
    Lx = sim_params["Lx"]
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, sim_params["Ly"], Nx, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap('plasma')
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)

    cax = ax.imshow(frames[0].T, origin='lower', cmap=cmap,
                    extent=[0, Lx, 0, sim_params["Ly"]],
                    vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(cax, label='Density', ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    subtitle = f"Nx={sim_params['Nx']}, Lx={sim_params['Lx']}, g={sim_params['g']}, ν={sim_params['nu']}, κ={sim_params['kappa']}"
    title = ax.set_title("Rayleigh–Taylor Instability\n" + subtitle)

    def update(frame_idx):
        cax.set_data(frames[frame_idx].T)
        title.set_text(f"RTI Density Field\n{subtitle}\nTime step: {frame_idx}")
        return cax, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=0, blit=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
