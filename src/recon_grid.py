import numpy as np


def grid_reconstruct(operator, samples, trajectory, maps):
    # Polar integration uses the Jacobian |k|. Give DC the average
    # radius of its integration cell so it contributes finite weight.
    radius = np.linalg.norm(trajectory.reshape(-1, 2), axis=1)
    spacing = np.pi / operator.size
    weights = np.maximum(radius, spacing / 4)
    coils = operator.adjoint(samples * weights)
    return np.sum(maps.conj() * coils, axis=0)
