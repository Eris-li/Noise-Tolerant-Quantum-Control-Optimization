from __future__ import annotations

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.analysis.noise_tolerant_rydberg import (
    blockade_branch_hamiltonians,
    detuning_noise_hamiltonians,
    propagate_blockade_branches,
    propagate_first_order_sensitivity,
)


class NoiseTolerantRydbergSensitivityTest(unittest.TestCase):
    def test_common_detuning_shifts_both_blockaded_single_rydberg_states(self) -> None:
        _h10, _h01, h11 = blockade_branch_hamiltonians(0.2, detuning=0.37)
        self.assertAlmostEqual(float(h11[1, 1].real), 0.37)
        self.assertAlmostEqual(float(h11[2, 2].real), 0.37)

    def test_atom_one_detuning_noise_matches_w_basis_perturbation(self) -> None:
        h10, h01, h11 = detuning_noise_hamiltonians(detuning_1_sign=1.0, detuning_2_sign=0.0)

        self.assertAlmostEqual(float(h10[1, 1].real), 1.0)
        self.assertAlmostEqual(float(h01[1, 1].real), 0.0)
        self.assertAlmostEqual(float(h11[1, 1].real), 0.5)
        self.assertAlmostEqual(float(h11[2, 2].real), 0.5)
        self.assertAlmostEqual(float(h11[1, 2].real), 0.5)
        self.assertAlmostEqual(float(h11[2, 1].real), 0.5)

    def test_amplitude_sensitivity_matches_finite_difference(self) -> None:
        phases = np.array([0.1, 0.7, -0.2, 1.1], dtype=np.float64)
        total_time = 3.7
        step = 1e-6

        nominal, first_order = propagate_first_order_sensitivity(
            phases,
            total_time=total_time,
            noise_kind="amplitude",
        )
        plus = propagate_blockade_branches(phases, total_time=total_time, amplitude_error=step)
        minus = propagate_blockade_branches(phases, total_time=total_time, amplitude_error=-step)

        for nominal_branch, plus_branch, minus_branch, sensitivity_branch in zip(
            nominal,
            plus,
            minus,
            first_order,
            strict=True,
        ):
            np.testing.assert_allclose(nominal_branch, (plus_branch + minus_branch) / 2.0, atol=1e-9, rtol=1e-9)
            finite_difference = (plus_branch - minus_branch) / (2.0 * step)
            np.testing.assert_allclose(sensitivity_branch, finite_difference, atol=2e-6, rtol=2e-6)

    def test_sign_reversed_detuning_sensitivity_matches_finite_difference(self) -> None:
        phases = np.array([0.1, 0.7, -0.2, 1.1], dtype=np.float64)
        total_time = 3.7
        signs = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        step = 1e-6

        nominal, first_order = propagate_first_order_sensitivity(
            phases,
            total_time=total_time,
            noise_kind="detuning",
            detuning_sign_trace=signs,
        )
        plus = propagate_blockade_branches(
            phases,
            total_time=total_time,
            detuning_trace=step * signs,
        )
        minus = propagate_blockade_branches(
            phases,
            total_time=total_time,
            detuning_trace=-step * signs,
        )

        for nominal_branch, plus_branch, minus_branch, sensitivity_branch in zip(
            nominal,
            plus,
            minus,
            first_order,
            strict=True,
        ):
            np.testing.assert_allclose(nominal_branch, (plus_branch + minus_branch) / 2.0, atol=1e-9, rtol=1e-9)
            finite_difference = (plus_branch - minus_branch) / (2.0 * step)
            np.testing.assert_allclose(sensitivity_branch, finite_difference, atol=2e-6, rtol=2e-6)

    def test_atom_one_detuning_sensitivity_matches_finite_difference(self) -> None:
        phases = np.array([0.1, 0.7, -0.2, 1.1], dtype=np.float64)
        total_time = 3.7
        signs = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        zeros = np.zeros_like(signs)
        step = 1e-6

        nominal, first_order = propagate_first_order_sensitivity(
            phases,
            total_time=total_time,
            noise_kind="detuning",
            detuning_1_sign_trace=signs,
            detuning_2_sign_trace=zeros,
        )
        plus = propagate_blockade_branches(
            phases,
            total_time=total_time,
            detuning_1_trace=step * signs,
            detuning_2_trace=zeros,
        )
        minus = propagate_blockade_branches(
            phases,
            total_time=total_time,
            detuning_1_trace=-step * signs,
            detuning_2_trace=zeros,
        )

        for nominal_branch, plus_branch, minus_branch, sensitivity_branch in zip(
            nominal,
            plus,
            minus,
            first_order,
            strict=True,
        ):
            np.testing.assert_allclose(nominal_branch, (plus_branch + minus_branch) / 2.0, atol=1e-9, rtol=1e-9)
            finite_difference = (plus_branch - minus_branch) / (2.0 * step)
            np.testing.assert_allclose(sensitivity_branch, finite_difference, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
