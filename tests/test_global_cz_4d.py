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
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


class GlobalCZ4DTest(unittest.TestCase):
    def test_dimension_and_initial_state(self) -> None:
        model = GlobalCZ4DModel(species=idealised_yb171())
        self.assertEqual(model.dimension(), 4)
        self.assertEqual(model.initial_state().shape[0], 4)

    def test_fidelity_bounds(self) -> None:
        model = GlobalCZ4DModel(species=idealised_yb171())
        fidelity = model.phase_gate_fidelity(model.initial_state().full().ravel(), 0.0)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)

    def test_optimizer_objective_runs(self) -> None:
        model = GlobalCZ4DModel(species=idealised_yb171())
        optimizer = PaperGlobalPhaseOptimizer(
            model,
            GlobalPhaseOptimizationConfig(num_tslots=6, evo_time=2.0, max_iter=1),
        )
        variables = list(np.asarray(optimizer.initial_phases()).ravel()) + [0.0]
        objective, gradient = optimizer.objective_and_gradient(np.asarray(variables, dtype=float))
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (7,))


if __name__ == "__main__":
    unittest.main()
