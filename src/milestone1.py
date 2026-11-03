"""Generate the M1 figures with the versioned default configuration."""
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.phantom import breathing, phantom
from src.trajectory import radial


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/default.toml").open("rb") as stream:
        config = tomllib.load(stream)
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    time = np.arange(config["spokes"]) * config["tr"]
    displacement = breathing(time, config["period"], config["amplitude"], config["seed"])
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for axis, shift, label in zip(axes[:2], (0, displacement.max()), ("Expiration", "Peak inspiration")):
        axis.imshow(phantom(config["size"], shift), cmap="gray", vmin=0, vmax=1)
        axis.set_title(label)
        axis.axis("off")
    axes[2].plot(time, displacement)
    axes[2].set(xlabel="Time (s)", ylabel="Inferior displacement (pixels)", title="Simulated breathing")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"F1_phantom.{extension}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 5), layout="constrained")
    for ky, kx in radial(config["size"], 30).transpose(0, 2, 1):
        axis.plot(kx, ky, linewidth=.8)
    axis.set(aspect="equal", xlabel="kx (rad/pixel)", ylabel="ky (rad/pixel)",
             title="First 30 golden-angle spokes")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"F2_trajectory.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
