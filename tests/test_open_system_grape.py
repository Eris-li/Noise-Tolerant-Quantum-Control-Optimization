from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import (
    OpenSystemGRAPEConfig,
    OpenSystemGRAPEResult,
    OpenSystemGRAPEOptimizer,
)


class OpenSystemGRAPETest(unittest.TestCase):
    def test_single_restart_uses_mocked_qtrl_backend(self) -> None:
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
        fake_final_amps = np.column_stack(
            [
                np.linspace(0.2, -0.2, optimizer.config.num_tslots, dtype=np.float64),
                np.linspace(-0.1, 0.1, optimizer.config.num_tslots, dtype=np.float64),
            ]
        )
        fake_result = SimpleNamespace(
            final_amps=fake_final_amps,
            fid_err=0.25,
            num_iter=1,
            num_fid_func_calls=2,
            termination_reason="mocked optimize_pulse result",
        )
        packaged_result = OpenSystemGRAPEResult(
            ctrl_x=fake_final_amps[:, 0],
            ctrl_y=fake_final_amps[:, 1],
            amplitudes=np.linalg.norm(fake_final_amps, axis=1),
            phases=np.mod(np.arctan2(fake_final_amps[:, 1], fake_final_amps[:, 0]), 2.0 * np.pi),
            target_theta=0.0,
            optimized_theta=0.1,
            fid_err=0.25,
            probe_fidelity=0.9,
            num_iter=1,
            num_fid_func_calls=2,
            wall_time=0.01,
            termination_reason="mocked optimize_pulse result",
            evo_time=optimizer.config.evo_time,
            num_tslots=optimizer.config.num_tslots,
        )
        with patch(
            "neutral_yb.optimization.open_system_grape.pulseoptim.optimize_pulse",
            return_value=fake_result,
        ) as optimize_pulse, patch.object(
            optimizer,
            "_result_from_controls",
            return_value=packaged_result,
        ) as result_from_controls:
            result = optimizer.optimize()

        optimize_pulse.assert_called_once()
        result_from_controls.assert_called_once()
        self.assertEqual(result.ctrl_x.shape[0], 6)
        self.assertEqual(result.ctrl_y.shape[0], 6)
        self.assertAlmostEqual(result.fid_err, 0.25)
        self.assertEqual(result.num_iter, 1)
        self.assertEqual(result.num_fid_func_calls, 2)
        self.assertGreaterEqual(result.probe_fidelity, 0.0)
        self.assertLessEqual(result.probe_fidelity, 1.0)
