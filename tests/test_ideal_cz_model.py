from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.ideal_cz import IdealCZModel
from neutral_yb.optimization.phase_grape import PhaseOnlyGrapeConfig, PhaseOnlyGrapeOptimizer


class IdealCZModelTest(unittest.TestCase):
    def test_model_dimensions(self) -> None:
        model = IdealCZModel(species=idealised_yb171())
        self.assertEqual(model.dimension(), 7)
        self.assertEqual(model.computational_projector().shape, (7, 4))

    def test_optimizer_objective_runs(self) -> None:
        model = IdealCZModel(species=idealised_yb171())
        optimizer = PhaseOnlyGrapeOptimizer(model, PhaseOnlyGrapeConfig(num_tslots=4, evo_time=1.0, max_iter=1))
        objective, gradient = optimizer.objective_and_gradient(optimizer.initial_phases())
        self.assertTrue(objective >= 0.0)
        self.assertEqual(gradient.shape, (4,))


if __name__ == "__main__":
    unittest.main()
