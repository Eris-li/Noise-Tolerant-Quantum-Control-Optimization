from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.finite_blockade_cz_5d import FiniteBlockadeCZ5DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


class FiniteBlockadeCZ5DTest(unittest.TestCase):
    def test_dimension_and_initial_state(self) -> None:
        model = FiniteBlockadeCZ5DModel(
            species=idealised_yb171(),
            blockade_shift=8.0,
            static_detuning_01=0.01,
            static_detuning_11=0.01,
            rabi_scale=0.98,
        )
        self.assertEqual(model.dimension(), 5)
        self.assertEqual(model.initial_state().shape[0], 5)

    def test_optimizer_objective_runs(self) -> None:
        model = FiniteBlockadeCZ5DModel(
            species=idealised_yb171(),
            blockade_shift=8.0,
            static_detuning_01=0.01,
            static_detuning_11=0.01,
            rabi_scale=0.98,
        )
        optimizer = PaperGlobalPhaseOptimizer(
            model,
            GlobalPhaseOptimizationConfig(num_tslots=6, evo_time=2.0, max_iter=1),
        )
        variables = list(optimizer.initial_phases()) + [0.0]
        objective, gradient = optimizer.objective_and_gradient(variables)
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (7,))


if __name__ == "__main__":
    unittest.main()
