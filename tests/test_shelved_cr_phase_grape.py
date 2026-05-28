from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.analysis.uv_edge_scan import (
    UVDenseEdgeScanConfig,
    dense_time_is_allowed,
    summarize_uv_edge_rows,
    write_uv_edge_artifacts,
    load_uv_edge_artifacts,
)
from neutral_yb.models.ma2023_pulse import gaussian_edge_envelope_from_times
from neutral_yb.optimization.shelved_cr_phase_grape import (
    RydbergDecayShelvedCRPhaseGRAPE,
    ShelvedCRPhaseGRAPEConfig,
    resample_phase_controls,
)


class ShelvedCRPhaseGRAPETest(unittest.TestCase):
    def build_optimizer(self) -> RydbergDecayShelvedCRPhaseGRAPE:
        return RydbergDecayShelvedCRPhaseGRAPE(
            ShelvedCRPhaseGRAPEConfig(
                omega_max_mhz=10.0,
                total_time_ns=80.0,
                edge_time_ns=10.0,
                num_tslots=5,
                smoothness_weight=1e-4,
                curvature_weight=1e-4,
                rydberg_lifetime_s=65e-6,
            )
        )

    def test_gaussian_edge_uses_physical_edge_time(self) -> None:
        optimizer = self.build_optimizer()
        expected = gaussian_edge_envelope_from_times(5, 80.0, 10.0)
        self.assertTrue(np.allclose(optimizer.envelope, expected))
        self.assertAlmostEqual(float(optimizer.envelope[0]), 0.0)
        self.assertAlmostEqual(float(optimizer.envelope[-1]), 0.0)

    def test_phase_gradient_matches_finite_difference(self) -> None:
        optimizer = self.build_optimizer()
        variables = optimizer.initial_guess(123)
        _, gradient = optimizer.objective_and_gradient(variables)

        index = 2
        step = 1e-6
        shifted_plus = np.array(variables, copy=True)
        shifted_minus = np.array(variables, copy=True)
        shifted_plus[index] += step
        shifted_minus[index] -= step
        objective_plus, _ = optimizer.objective_and_gradient(shifted_plus)
        objective_minus, _ = optimizer.objective_and_gradient(shifted_minus)
        finite_difference = (objective_plus - objective_minus) / (2.0 * step)
        self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=5)

    def test_resample_phase_controls_preserves_slot_count(self) -> None:
        phases = [0.0, np.pi / 2.0, np.pi]
        resampled = resample_phase_controls(phases, 7)
        self.assertEqual(resampled.shape, (7,))

    def test_uv_edge_summary_and_io_round_trip(self) -> None:
        rows = [
            {"omega_max_mhz": 10.0, "edge_ns": 0.0, "total_time_ns": 60.0, "fidelity": 0.90, "passed": False, "phases": [0.0]},
            {"omega_max_mhz": 10.0, "edge_ns": 0.0, "total_time_ns": 80.0, "fidelity": 0.9995, "passed": True, "phases": [0.1]},
        ]
        summary, selected = summarize_uv_edge_rows(rows)
        self.assertEqual(summary[0]["shortest_passing_time_ns"], 80.0)
        self.assertEqual(selected[0]["selection_kind"], "first above threshold")
        self.assertFalse(dense_time_is_allowed(60.0, 30.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            config = UVDenseEdgeScanConfig(output_dir=Path(temp_dir), max_iter=1, num_tslots=2)
            write_uv_edge_artifacts(config, rows, summary, selected)
            loaded_rows, loaded_summary, loaded_selected = load_uv_edge_artifacts(config)
        self.assertEqual(loaded_rows[0]["fidelity"], rows[0]["fidelity"])
        self.assertEqual(loaded_summary[0]["best_time_ns"], summary[0]["best_time_ns"])
        self.assertEqual(loaded_selected[0]["selection_kind"], selected[0]["selection_kind"])


if __name__ == "__main__":
    unittest.main()
