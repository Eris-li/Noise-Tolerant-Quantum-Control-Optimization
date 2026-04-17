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


class TwoPhotonCZOpen10DTest(unittest.TestCase):
    def build_model(self) -> TwoPhotonCZOpen10DModel:
        return TwoPhotonCZOpen10DModel(
            species=idealised_yb171(),
            lower_rabi=4.0,
            upper_rabi=4.0,
            intermediate_detuning=8.0,
            blockade_shift=10.0,
            two_photon_detuning_01=0.01,
            two_photon_detuning_11=0.01,
            noise=TwoPhotonOpenNoiseConfig(
                intermediate_decay_rate=0.02,
                rydberg_decay_rate=0.01,
                rydberg_dephasing_rate=0.005,
                common_two_photon_detuning=0.003,
            ),
        )

    def test_dimension_and_collapse_ops(self) -> None:
        model = self.build_model()
        self.assertEqual(model.dimension(), 10)
        self.assertGreater(len(model.collapse_operators()), 0)
        self.assertEqual(model.drift_liouvillian().shape, (100, 100))

    def test_probe_fidelity_bounds(self) -> None:
        model = self.build_model()
        optimizer = OpenSystemGRAPEOptimizer(
            model=model,
            config=OpenSystemGRAPEConfig(num_tslots=8, evo_time=2.0, max_iter=1, num_restarts=1),
        )
        ctrl_x = np.zeros(8, dtype=np.float64)
        ctrl_y = np.zeros(8, dtype=np.float64)
        states = optimizer.evolve_probe_states(ctrl_x, ctrl_y)
        theta, fidelity = model.optimize_theta_for_probe_states(states)
        self.assertGreaterEqual(theta, 0.0)
        self.assertLessEqual(theta, 2.0 * np.pi)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)
