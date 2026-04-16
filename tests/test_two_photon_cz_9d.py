from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.linear_control_grape import (
    LinearControlGRAPEOptimizer,
    LinearControlOptimizationConfig,
)


class TwoPhotonCZ9DTest(unittest.TestCase):
    def test_dimension_and_control_channels(self) -> None:
        model = TwoPhotonCZ9DModel(
            species=idealised_yb171(),
            lower_rabi=4.0,
            upper_rabi=4.0,
            intermediate_detuning=8.0,
            blockade_shift=9.0,
        )
        self.assertEqual(model.dimension(), 9)
        self.assertEqual(len(model.control_hamiltonians()), 2)
        self.assertEqual(model.initial_state().shape[0], 9)

    def test_optimizer_objective_runs_for_two_controls(self) -> None:
        model = TwoPhotonCZ9DModel(
            species=idealised_yb171(),
            lower_rabi=4.0,
            upper_rabi=4.0,
            intermediate_detuning=8.0,
            blockade_shift=9.0,
            two_photon_detuning_01=0.02,
            two_photon_detuning_11=0.02,
        )
        optimizer = LinearControlGRAPEOptimizer(
            model,
            LinearControlOptimizationConfig(num_tslots=5, evo_time=2.0, max_iter=1),
        )
        control_matrix = optimizer.initial_controls()
        variables = list(np.asarray(control_matrix).ravel()) + [0.0]
        objective, gradient = optimizer.objective_and_gradient(np.asarray(variables, dtype=float))
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (11,))


if __name__ == "__main__":
    unittest.main()
