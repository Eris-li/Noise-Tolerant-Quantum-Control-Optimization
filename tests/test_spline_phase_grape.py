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
from neutral_yb.optimization.spline_phase_grape import (
    SplinePhaseOptimizationConfig,
    SplinePhaseOptimizer,
)


class SplinePhaseOptimizerTest(unittest.TestCase):
    def test_interpolation_shape_and_objective(self) -> None:
        model = TwoPhotonCZ9DModel(
            species=idealised_yb171(),
            lower_rabi=4.0,
            upper_rabi=4.0,
            intermediate_detuning=8.0,
            blockade_shift=9.0,
        )
        optimizer = SplinePhaseOptimizer(
            model,
            SplinePhaseOptimizationConfig(num_tslots=20, num_nodes=6, evo_time=2.0, max_iter=1),
        )
        self.assertEqual(optimizer.interpolation_matrix.shape, (20, 6))
        nodes = optimizer.initial_nodes()
        slices = optimizer.slice_phases_from_nodes(nodes)
        self.assertEqual(slices.shape, (20,))
        variables = np.concatenate([nodes, np.array([0.0])])
        objective, gradient = optimizer.objective_and_gradient(variables)
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (7,))


if __name__ == "__main__":
    unittest.main()
