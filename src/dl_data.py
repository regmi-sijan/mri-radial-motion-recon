"""Generate paired gridded inputs and simulated respiratory-state targets."""
import numpy as np

from src.navigator import navigate
from src.nufft_ops import NUFFT
from src.phantom import phantom
from src.recon_grid import grid_coils
from src.simulate import simulate


def case(config, seed):
    rng = np.random.default_rng(seed)
    settings = dict(config, case_seed=seed, seed=seed, period=rng.uniform(3.5, 4.5),
                    amplitude=rng.uniform(5, 10), snr=rng.uniform(15, 60))
    _, motion, maps, trajectory, samples = simulate(settings)
    _, bins = navigate(samples, settings["tr"], settings["bins"])
    size = settings["size"]
    coil_images, targets = [], []
    for indices in bins:
        path = trajectory[indices]
        data = samples[:, indices].reshape(len(maps), -1)
        coil_images.append(grid_coils(NUFFT(size, path), data, path) / len(indices))
        targets.append(phantom(size, motion[indices].mean(), seed))
    coil_images = np.asarray(coil_images)
    baseline = abs(np.sum(coil_images * maps.conj(), axis=1))
    # Estimate gain from a separate, unit-amplitude simulated point source.
    # For the centered pixel, all forward FFT samples are 1/grid.
    radius = np.linalg.norm(trajectory, axis=-1)
    gain = np.maximum(radius, np.pi / (4 * size)).sum() / (len(trajectory) * (4 * size)**2)
    coil_images /= gain
    baseline /= gain
    channels = np.concatenate((coil_images.real, coil_images.imag), axis=1)
    return tuple(np.asarray(x, dtype=np.float32) for x in
                 (channels.reshape(-1, size, size), baseline, targets))
