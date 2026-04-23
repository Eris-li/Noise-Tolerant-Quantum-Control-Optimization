from __future__ import annotations

import unittest

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    yb171_experimental_calibration,
)


class Yb171CalibrationTests(unittest.TestCase):
    def test_physical_time_round_trip(self) -> None:
        calibration = yb171_experimental_calibration()
        gate_time_s = 136e-9
        omega_max_hz = calibration.effective_rabi_hz_max
        t_omega = calibration.physical_gate_time_to_dimensionless(
            gate_time_s,
            effective_rabi_hz=omega_max_hz,
        )
        reconstructed = calibration.dimensionless_gate_time_to_seconds(
            t_omega,
            effective_rabi_hz=omega_max_hz,
        )
        self.assertAlmostEqual(reconstructed, gate_time_s, places=15)

    def test_model_builder_accepts_effective_rabi_override(self) -> None:
        default_model = build_yb171_v4_calibrated_model()
        fast_model = build_yb171_v4_calibrated_model(
            effective_rabi_hz=12e6,
        )
        self.assertAlmostEqual(default_model.uv_rabi, 1.0)
        self.assertAlmostEqual(fast_model.uv_rabi, 1.0)
        self.assertNotAlmostEqual(default_model.blockade_shift, fast_model.blockade_shift)
        self.assertGreater(default_model.clock_pi_time, 0.0)
        self.assertEqual(default_model.clock_num_steps, yb171_experimental_calibration().clock_num_steps)

    def test_quasistatic_ensemble_is_reproducible_and_differs_from_nominal(self) -> None:
        nominal = build_yb171_v4_calibrated_model()
        ensemble_a = build_yb171_v4_quasistatic_ensemble(ensemble_size=3, seed=123)
        ensemble_b = build_yb171_v4_quasistatic_ensemble(ensemble_size=3, seed=123)

        self.assertEqual(len(ensemble_a), 3)
        self.assertEqual(len(ensemble_b), 3)
        for model_a, model_b in zip(ensemble_a, ensemble_b):
            self.assertAlmostEqual(model_a.noise.common_uv_detuning, model_b.noise.common_uv_detuning)
            self.assertAlmostEqual(model_a.noise.uv_amplitude_scale, model_b.noise.uv_amplitude_scale)
            self.assertAlmostEqual(model_a.noise.blockade_shift_offset, model_b.noise.blockade_shift_offset)
            self.assertAlmostEqual(model_a.noise.common_clock_detuning, model_b.noise.common_clock_detuning)
            self.assertEqual(model_a.noise.clock_phase_trace_prefix, model_b.noise.clock_phase_trace_prefix)

        different_from_nominal = any(
            abs(member.noise.common_uv_detuning - nominal.noise.common_uv_detuning) > 1e-12
            or abs(member.noise.uv_amplitude_scale - nominal.noise.uv_amplitude_scale) > 1e-12
            or abs(member.noise.blockade_shift_offset - nominal.noise.blockade_shift_offset) > 1e-12
            for member in ensemble_a
        )
        self.assertTrue(different_from_nominal)

    def test_default_v4_noise_matches_current_yb171_assumptions(self) -> None:
        model = build_yb171_v4_calibrated_model()
        self.assertEqual(model.noise.common_clock_detuning, 0.0)
        self.assertEqual(model.noise.common_uv_detuning, 0.0)
        self.assertEqual(model.noise.rydberg_dephasing_rate, 0.0)
        self.assertEqual(model.noise.clock_decay_rate, 0.0)
        self.assertGreater(model.noise.clock_scattering_rate, 0.0)
        self.assertGreater(model.noise.clock_loss_rate, 0.0)

    def test_strict_literature_profile_disables_surrogate_terms(self) -> None:
        calibration = yb171_experimental_calibration(profile="strict_literature_minimal")
        self.assertEqual(calibration.profile_name, "strict_literature_minimal")
        self.assertEqual(calibration.quasistatic_uv_detuning_rms_hz, 0.0)
        self.assertEqual(calibration.clock_phase_noise_psd_level_rad2_per_hz, 0.0)
        self.assertTrue(calibration.clock_decay_as_single_loss_channel)

        model = build_yb171_v4_calibrated_model(profile="strict_literature_minimal")
        self.assertGreater(model.noise.clock_decay_rate, 0.0)
        self.assertEqual(model.noise.clock_scattering_rate, 0.0)
        self.assertEqual(model.noise.clock_loss_rate, 0.0)

        ensemble = build_yb171_v4_quasistatic_ensemble(
            ensemble_size=1,
            seed=5,
            profile="strict_literature_minimal",
        )
        self.assertEqual(ensemble[0].noise.common_uv_detuning, 0.0)
        self.assertTrue(all(value == 0.0 for value in ensemble[0].noise.clock_phase_trace_prefix))


if __name__ == "__main__":
    unittest.main()
