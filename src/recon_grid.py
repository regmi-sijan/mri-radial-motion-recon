import numpy as np


def grid_coils(operator, samples, trajectory):
    # Polar integration uses the Jacobian |k|. Give DC the average
    # radius of its integration cell so it contributes finite weight.
    radius = np.linalg.norm(trajectory.reshape(-1, 2), axis=1)
    spacing = np.pi / operator.size
    weights = np.maximum(radius, spacing / 4)
    return operator.adjoint(samples * weights)


def grid_reconstruct(operator, samples, trajectory, maps):
    coils = grid_coils(operator, samples, trajectory)
    return np.sum(maps.conj() * coils, axis=0)
