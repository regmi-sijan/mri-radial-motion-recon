import numpy as np


def sensitivities(size, count):
    y, x = np.mgrid[:size, :size] * (2 / size) - 1
    angle = np.arange(count)[:, None, None] * (2 * np.pi / count)
    dx, dy = x - np.cos(angle), y - np.sin(angle)
    maps = np.exp(-(dx**2 + dy**2) / .9 + 1j * (dx * np.sin(angle) - dy * np.cos(angle)))
    return maps / np.sqrt(np.sum(abs(maps)**2, axis=0))
