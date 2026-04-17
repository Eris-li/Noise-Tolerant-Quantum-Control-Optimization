from __future__ import annotations

import unittest

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
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
        calibration = yb171_experimental_calibration()
        default_model = build_yb171_v4_calibrated_model()
        fast_model = build_yb171_v4_calibrated_model(
            effective_rabi_hz=calibration.effective_rabi_hz_max,
        )
        self.assertNotAlmostEqual(default_model.lower_rabi, fast_model.lower_rabi)
        self.assertNotAlmostEqual(default_model.intermediate_detuning, fast_model.intermediate_detuning)
        self.assertNotAlmostEqual(default_model.blockade_shift, fast_model.blockade_shift)


if __name__ == "__main__":
    unittest.main()
