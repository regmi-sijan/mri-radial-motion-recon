"""Coronal phantom. Positive displacement moves organs inferiorly."""
import numpy as np
from scipy.ndimage import gaussian_filter


def phantom(size, displacement=0, case_seed=None):
    y, x = np.mgrid[:size, :size] * (128 / size) - 64
    displacement *= 128 / size

    def ellipse(cx, cy, rx, ry):
        return ((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1

    rng = np.random.default_rng(case_seed)
    shift, width, height, intensity, lesion_x, lesion_y, radius = (
        (0, 21, 15, .61, -26, -7, 3.5) if case_seed is None else
        (rng.uniform(-3, 3), rng.uniform(18, 23), rng.uniform(13, 17),
         rng.uniform(.52, .70), rng.uniform(-31, -20), rng.uniform(-12, -4), rng.uniform(2.5, 4.5))
    )
    image = np.zeros((size, size))
    image[ellipse(0, 0, 51, 58)] = .85
    image[ellipse(0, 0, 47, 54)] = .35
    image[ellipse(-21, -29 + displacement, 22, 18)] = .07
    image[ellipse(4, 9, 5, 38)] = .94
    image[ellipse(-23 + shift, -8 + displacement, width, height - .12 * displacement)] = intensity
    image[ellipse(23, 12 + .4 * displacement, 9, 15)] = .49
    image[ellipse(lesion_x + shift, lesion_y + displacement, radius, radius)] = intensity * 1.3
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
