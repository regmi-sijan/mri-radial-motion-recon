"""Simulate a free-breathing acquisition with independent complex noise."""
import numpy as np

from src.coils import sensitivities
from src.nufft_ops import NUFFT
from src.phantom import breathing, phantom
from src.recon_grid import grid_reconstruct
from src.trajectory import radial


def simulate(config):
    size = config["size"]
    time = np.arange(config["spokes"]) * config["tr"]
    motion = breathing(time, config["period"], config["amplitude"], config["seed"])
    maps = sensitivities(size, config["coils"])
    trajectory = radial(size, len(time))
    samples = np.empty((len(maps), len(time), 2 * size), dtype=complex)
    chunk = config["chunk_spokes"]
    for start in range(0, len(time), chunk):
        subset = slice(start, start + chunk)
        operator = NUFFT(size, trajectory[subset])
        image = phantom(size, motion[subset].mean(), config.get("case_seed"))
        samples[:, subset] = operator.forward(maps * image).reshape(len(maps), -1, 2 * size)

    # Calibrate per-quadrature noise against a 402-spoke static reconstruction.
    reference = radial(size, config["closure_spokes"])
    operator = NUFFT(size, reference)
    rng = np.random.default_rng(config["seed"] + 1)

    def noise(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    signal = grid_reconstruct(operator, operator.forward(maps * phantom(size)), reference, maps)
    noise_image = grid_reconstruct(operator, noise((len(maps), reference.shape[0] * 2 * size)), reference, maps)
    row, col = np.mgrid[:size, :size] * (128 / size) - 64
    liver = (row + 8)**2 + (col + 16)**2 < 25
    sigma = abs(signal[liver]).mean() / (config["snr"] * noise_image.real[liver].std())
    return time, motion, maps, trajectory, samples + sigma * noise(samples.shape)
