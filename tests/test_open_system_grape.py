from __future__ import annotations

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


class OpenSystemGRAPETest(unittest.TestCase):
    def build_optimizer(self) -> OpenSystemGRAPEOptimizer:
        model = TwoPhotonCZOpen10DModel(
            species=idealised_yb171(),
            lower_rabi=4.0,
            upper_rabi=4.0,
            intermediate_detuning=8.0,
            blockade_shift=10.0,
            noise=TwoPhotonOpenNoiseConfig(
                intermediate_decay_rate=0.01,
                rydberg_decay_rate=0.005,
                rydberg_dephasing_rate=0.002,
            ),
        )
        return OpenSystemGRAPEOptimizer(
            model=model,
            config=OpenSystemGRAPEConfig(
                num_tslots=4,
                evo_time=1.5,
                max_iter=2,
                num_restarts=1,
                init_pulse_type="ZERO",
                control_smoothness_weight=0.0,
                control_curvature_weight=0.0,
            ),
        )

    def test_objective_runs(self) -> None:
        optimizer = self.build_optimizer()
        ctrl_x = np.array([0.05, -0.03, 0.02, -0.01], dtype=np.float64)
        ctrl_y = np.array([0.01, 0.04, -0.02, -0.03], dtype=np.float64)
        variables = np.concatenate([ctrl_x, ctrl_y, np.array([0.2])])
        objective, gradient = optimizer.objective_and_gradient(variables)
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (9,))

    def test_theta_gradient_matches_finite_difference(self) -> None:
        optimizer = self.build_optimizer()
        ctrl_x = np.array([0.02, -0.01, 0.03, -0.02], dtype=np.float64)
        ctrl_y = np.array([-0.01, 0.03, -0.02, 0.01], dtype=np.float64)
        variables = np.concatenate([ctrl_x, ctrl_y, np.array([0.4])])
        _, gradient = optimizer.objective_and_gradient(variables)

        step = 1e-6
        shifted_plus = np.array(variables, copy=True)
        shifted_minus = np.array(variables, copy=True)
        shifted_plus[-1] += step
        shifted_minus[-1] -= step
        objective_plus, _ = optimizer.objective_and_gradient(shifted_plus)
        objective_minus, _ = optimizer.objective_and_gradient(shifted_minus)
        finite_difference = (objective_plus - objective_minus) / (2.0 * step)
        self.assertAlmostEqual(gradient[-1], finite_difference, places=5)

    def test_single_restart_runs(self) -> None:
        optimizer = self.build_optimizer()
        result = optimizer.optimize()
        self.assertEqual(result.ctrl_x.shape[0], 4)
        self.assertEqual(result.ctrl_y.shape[0], 4)
        self.assertGreaterEqual(result.probe_fidelity, 0.0)
        self.assertLessEqual(result.probe_fidelity, 1.0)


if __name__ == "__main__":
    unittest.main()
