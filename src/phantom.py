"""Coronal phantom. Positive displacement moves organs inferiorly."""
import numpy as np
from scipy.ndimage import gaussian_filter


def phantom(size, displacement=0):
    y, x = np.mgrid[:size, :size] * (128 / size) - 64
    displacement *= 128 / size

    def ellipse(cx, cy, rx, ry):
        return ((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1

    image = np.zeros((size, size))
    image[ellipse(0, 0, 51, 58)] = .85
    image[ellipse(0, 0, 47, 54)] = .35
    image[ellipse(-21, -29 + displacement, 22, 18)] = .07
    image[ellipse(4, 9, 5, 38)] = .94
    image[ellipse(-23, -8 + displacement, 21, 15 - .12 * displacement)] = .61
    image[ellipse(23, 12 + .4 * displacement, 9, 15)] = .49
    image[ellipse(-26, -7 + displacement, 3.5, 3.5)] = .61 * 1.3
    # Finite spatial resolution softens ideal discontinuities at organ edges.
    return gaussian_filter(image, .85 * size / 128)


def breathing(time, period, amplitude, seed):
    rng = np.random.default_rng(seed)
    displacement = np.zeros_like(time)
    start = 0.
    while start <= time.max():
        duration = period * rng.uniform(.9, 1.1)
        height = amplitude * rng.uniform(.85, 1.15)
        cycle = (time >= start) & (time < start + duration)
        displacement[cycle] = height * np.sin(np.pi * (time[cycle] - start) / duration)**4
        start += duration
    return displacement
