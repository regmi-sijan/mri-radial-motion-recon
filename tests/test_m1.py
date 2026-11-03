from pathlib import Path
import tomllib
import unittest

import numpy as np

from src.coils import sensitivities
from src.nufft_ops import NUFFT
from src.phantom import phantom
from src.recon_grid import grid_reconstruct
from src.trajectory import radial


class TestM1(unittest.TestCase):
    def test_adjoint(self):
        rng = np.random.default_rng(42)

        def complex_normal(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        operator = NUFFT(32, radial(32, 31))
        image = complex_normal((3, 32, 32))
        data = complex_normal((3, 31 * 64))
        left = np.vdot(operator.forward(image), data)
        right = np.vdot(image, operator.adjoint(data))
        error = abs(left - right) / max(abs(left), abs(right))
        print(f"Adjoint relative error = {error:.3g}")
        self.assertLess(error, 1e-4)

    def test_static_closure(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/default.toml"
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        size = config["size"]
        trajectory = radial(size, config["closure_spokes"])
        maps = sensitivities(size, config["coils"])
        truth = phantom(size)
        operator = NUFFT(size, trajectory)
        samples = operator.forward(maps * truth)
        estimate = abs(grid_reconstruct(operator, samples, trajectory, maps))
        scale = np.vdot(estimate, truth) / np.vdot(estimate, estimate)
        error = np.linalg.norm(scale * estimate - truth) / np.linalg.norm(truth)
        print(f"Static closure NRMSE = {error:.4%}")
        self.assertLess(error, .05)
