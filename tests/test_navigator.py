from pathlib import Path
import tomllib
import unittest

import numpy as np

from src.navigator import navigate
from src.simulate import simulate


class TestNavigator(unittest.TestCase):
    def test_default_acquisition(self):
        path = Path(__file__).resolve().parents[1] / "configs/default.toml"
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        _, motion, _, _, samples = simulate(config)
        signal, bins = navigate(samples, config["tr"], config["bins"])
        correlation = np.corrcoef(signal, motion)[0, 1]
        self.assertGreater(correlation, .9)
        self.assertTrue(np.all(np.diff([motion[indices].mean() for indices in bins]) > 0))
        np.testing.assert_array_equal(np.sort(np.concatenate(bins)), np.arange(len(motion)))
