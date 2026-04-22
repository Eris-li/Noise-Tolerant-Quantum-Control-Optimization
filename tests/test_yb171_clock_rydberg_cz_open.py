from __future__ import annotations

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.yb171_clock_rydberg_cz_open import (
    Yb171ClockRydbergCZOpenModel,
    Yb171ClockRydbergNoiseConfig,
)


class Yb171ClockRydbergCZOpenModelTest(unittest.TestCase):
    def build_model(self) -> Yb171ClockRydbergCZOpenModel:
        return Yb171ClockRydbergCZOpenModel(
            species=idealised_yb171(),
            uv_rabi=1.0,
            blockade_shift=16.0,
            clock_pi_time=5.0,
            clock_num_steps=8,
            noise=Yb171ClockRydbergNoiseConfig(
                common_clock_detuning=0.01,
                common_uv_detuning=0.02,
                clock_decay_rate=1e-4,
                rydberg_decay_rate=0.01,
                neighboring_mf_leakage_rate=0.002,
            ),
        )

    def test_dimension_and_collapse_ops(self) -> None:
        model = self.build_model()
        self.assertEqual(model.dimension(), 11)
        self.assertEqual(model.active_gate_indices(), (0, 3))
        self.assertEqual(model.loss_index(), 10)
        self.assertGreater(len(model.collapse_operators()), 0)
        self.assertEqual(model.drift_liouvillian().shape, (121, 121))

    def test_clock_segments_are_present(self) -> None:
        model = self.build_model()
        segments = model.clock_segment_controls()
        self.assertEqual(len(segments["prefix_x"]), model.clock_num_steps)
        self.assertEqual(len(segments["suffix_x"]), model.clock_num_steps)
        self.assertGreater(np.max(segments["prefix_x"]), 0.0)
        self.assertAlmostEqual(float(segments["prefix_dt"]) * model.clock_num_steps, model.clock_pi_time)

    def test_phase_gate_fidelity_bounds(self) -> None:
        model = self.build_model()
        state = model.special_phase_gate_state().full().ravel()
        theta, fidelity = model.optimize_theta_for_ket(state)
        self.assertGreaterEqual(theta, 0.0)
        self.assertLessEqual(theta, 2.0 * np.pi)
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)

    def test_control_polar_conversion(self) -> None:
        model = self.build_model()
        amplitudes, phases = model.control_cartesian_to_polar(
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        self.assertTrue(np.allclose(amplitudes, np.array([1.0, 1.0, 1.0])))
        self.assertTrue(np.all(phases >= 0.0))
        self.assertTrue(np.all(phases <= 2.0 * np.pi))


if __name__ == "__main__":
    unittest.main()
