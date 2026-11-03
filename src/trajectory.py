import numpy as np


def radial(size, spokes):
    """Return (ky, kx) in radians per pixel, with exact DC on each spoke."""
    angle = np.deg2rad(np.arange(spokes) * 111.246 % 360)
    radius = np.arange(-size, size) * (np.pi / size)
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1)[:, None, :] * radius[None, :, None]
