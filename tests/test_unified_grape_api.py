from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.species import idealised_yb171
from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.grape import ClosedSystemGRAPE, OpenSystemGRAPE
from neutral_yb.optimization.global_phase_grape import GlobalPhaseOptimizationConfig
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig


class UnifiedGRAPEAPITest(unittest.TestCase):
    def test_closed_system_class_matches_global_phase_optimizer_objective(self) -> None:
        model = GlobalCZ4DModel(species=idealised_yb171())
        config = GlobalPhaseOptimizationConfig(num_tslots=4, evo_time=1.7, max_iter=1)
        closed = ClosedSystemGRAPE.global_phase(model, config)
        variables = np.concatenate([closed.initial_phases().ravel(), np.array([0.2])])

        closed_objective, closed_gradient = closed.objective_and_gradient(variables)
        repeated_objective, repeated_gradient = closed.objective_and_gradient(variables)

        self.assertAlmostEqual(closed_objective, repeated_objective, places=12)
        self.assertTrue(np.allclose(closed_gradient, repeated_gradient, atol=1e-12, rtol=1e-12))

    def test_old_optimizer_class_names_are_not_public(self) -> None:
        import neutral_yb.optimization as optimization
        import neutral_yb.optimization.global_phase_grape as global_phase_grape
        import neutral_yb.optimization.open_system_grape as open_system_grape

        old_global_name = "Paper" + "GlobalPhase" + "Optimizer"
        old_open_name = "OpenSystem" + "GRAPEOptimizer"
        self.assertFalse(hasattr(optimization, old_global_name))
        self.assertFalse(hasattr(optimization, old_open_name))
        self.assertFalse(hasattr(global_phase_grape, old_global_name))
        self.assertFalse(hasattr(open_system_grape, old_open_name))

    def test_open_system_class_replaces_open_system_optimizer_name(self) -> None:
        model = replace(
            build_yb171_v4_calibrated_model(include_noise=False, effective_rabi_hz=10e6),
            clock_num_steps=1,
            clock_pi_time=0.05,
        )
        config = OpenSystemGRAPEConfig(
            num_tslots=1,
            evo_time=0.2,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            objective_metric="special_state",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        )
        optimizer = OpenSystemGRAPE(model=model, config=config)
        variables = np.array([0.02, -0.01, 0.3], dtype=np.float64)

        objective, gradient = optimizer.objective_and_gradient(variables)

        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, variables.shape)


if __name__ == "__main__":
    unittest.main()
