"""Oversampled FFT with bilinear interpolation and its exact transpose."""
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


class NUFFT:
    def __init__(self, size, trajectory):
        self.size = size
        self.grid = 4 * size
        start = (self.grid - size) // 2
        self.crop = slice(start, start + size)
        coordinates = trajectory.reshape(-1, 2) * (self.grid / (2 * np.pi)) + self.grid // 2
        lower = np.floor(coordinates).astype(int)
        dy, dx = (coordinates - lower).T
        row, col = lower.T
        self.indices = np.stack([
            ((row + i) % self.grid) * self.grid + (col + j) % self.grid
            for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
        ])
        self.weights = np.stack([(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx])

    def forward(self, images):
        grid = np.zeros((len(images), self.grid, self.grid), dtype=complex)
        grid[:, self.crop, self.crop] = images
        axes = (-2, -1)
        spectrum = fftshift(fft2(ifftshift(grid, axes=axes), norm="ortho"), axes=axes)
        return np.sum(spectrum.reshape(len(images), -1)[:, self.indices] * self.weights, axis=1)

    def adjoint(self, samples):
        grids = []
        for coil in samples:
            values = (self.weights * coil).ravel()
            indices = self.indices.ravel()
            length = self.grid**2
            grids.append(np.bincount(indices, values.real, minlength=length)
                         + 1j * np.bincount(indices, values.imag, minlength=length))
        axes = (-2, -1)
        grids = np.asarray(grids).reshape(-1, self.grid, self.grid)
        images = fftshift(ifft2(ifftshift(grids, axes=axes), norm="ortho"), axes=axes)
        return images[:, self.crop, self.crop]
