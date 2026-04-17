from __future__ import annotations

import unittest

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


class OpenSystemGRAPETest(unittest.TestCase):
    def test_single_restart_runs(self) -> None:
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
        optimizer = OpenSystemGRAPEOptimizer(
            model=model,
            config=OpenSystemGRAPEConfig(
                num_tslots=6,
                evo_time=1.5,
                max_iter=1,
                max_wall_time=30.0,
                num_restarts=1,
            ),
        )
        result = optimizer.optimize()
        self.assertEqual(result.ctrl_x.shape[0], 6)
        self.assertEqual(result.ctrl_y.shape[0], 6)
        self.assertGreaterEqual(result.probe_fidelity, 0.0)
        self.assertLessEqual(result.probe_fidelity, 1.0)
