"""Generate the respiratory navigator and gridding comparison."""
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.navigator import navigate
from src.nufft_ops import NUFFT
from src.phantom import phantom
from src.recon_grid import grid_reconstruct
from src.simulate import simulate


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/default.toml").open("rb") as stream:
        config = tomllib.load(stream)
    time, motion, maps, trajectory, samples = simulate(config)
    signal, bins = navigate(samples, config["tr"], config["bins"])
    correlation = np.corrcoef(signal, motion)[0, 1]
    means = [motion[indices].mean() for indices in bins]
    print(f"Navigator r = {correlation:.4f}; bin means (pixels) = {np.round(means, 2)}")

    def reconstruct(indices):
        path = trajectory[indices]
        operator = NUFFT(config["size"], path)
        data = samples[:, indices].reshape(len(maps), -1)
        return abs(grid_reconstruct(operator, data, path, maps)) / len(path)

    averaged = reconstruct(np.arange(len(time)))
    images = [reconstruct(indices) for indices in bins]
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 3), layout="constrained")
    normalized = (signal - signal.mean()) / signal.std()
    axis.plot(time, (motion - motion.mean()) / motion.std(), label="True displacement")
    axis.plot(time, normalized, label=f"Navigator (r = {correlation:.3f})")
    for lower, upper in zip(bins, bins[1:]):
        axis.axhline((normalized[lower[-1]] + normalized[upper[0]]) / 2, color="gray", linestyle=":")
    axis.set(xlabel="Time (s)", ylabel="Standardized signal")
    axis.legend()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"F3_navigator.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), layout="constrained")
    # Use one display gain for all reconstructions, set from the averaged image.
    gain = .85 / np.percentile(averaged, 95)
    for row, index in zip(axes, (0, -1)):
        truth = phantom(config["size"], means[index])
        for axis, image, label in zip(row, (truth, gain * averaged, gain * images[index]),
                                      ("Bin truth", "All spokes (R1)", "Binned spokes (R2)")):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(label)
            axis.axis("off")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"M2_gridding.{ext}", dpi=180)


if __name__ == "__main__":
    main()
