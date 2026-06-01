from __future__ import annotations

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.evered2023_parallel_cz import (
    Evered2023DarkStateConfig,
    Evered2023ParallelCZCalibration,
    Evered2023TimeOptimalPulse,
    build_evered2023_ideal_global_cz_model,
    build_evered2023_two_photon_detuning_model,
    build_evered2023_two_photon_ladder_model,
)
from neutral_yb.models.evered2023_benchmarking import (
    diagonal_cz_average_gate_fidelity,
    evered2023_exponential_decay_fidelity_from_diagonal_map,
    repeated_diagonal_cz_average_fidelities,
)
from neutral_yb.optimization.grape import ClosedSystemGRAPE
from neutral_yb.optimization.global_phase_grape import GlobalPhaseOptimizationConfig
from neutral_yb.optimization.evered2023_parameterized_grape import (
    Evered2023ParameterizedGRAPEConfig,
)


class Evered2023ParallelCZTest(unittest.TestCase):
    def test_time_optimal_pulse_parameters_and_duration(self) -> None:
        pulse = Evered2023TimeOptimalPulse()
        self.assertAlmostEqual(pulse.amplitude_phase_modulation / (2.0 * np.pi), 0.1122)
        self.assertAlmostEqual(pulse.phase_rate, 1.0431)
        self.assertAlmostEqual(pulse.phase_offset, -0.7318)
        self.assertAlmostEqual(pulse.omega_t_over_2pi, 1.215)
        self.assertAlmostEqual(pulse.dimensionless_duration, 2.0 * np.pi * 1.215)
        self.assertAlmostEqual(
            pulse.physical_duration_seconds(4.6e6),
            1.215 / 4.6e6,
        )

    def test_time_optimal_pulse_matches_ideal_global_cz(self) -> None:
        pulse = Evered2023TimeOptimalPulse()
        model = build_evered2023_ideal_global_cz_model(species=idealised_yb171())
        optimizer = ClosedSystemGRAPE.global_phase(
            model=model,
            config=GlobalPhaseOptimizationConfig(
                num_tslots=200,
                evo_time=pulse.dimensionless_duration,
                max_iter=1,
            ),
        )
        phases = pulse.sampled_phases(optimizer.config.num_tslots)
        state = optimizer.final_state(phases)
        _theta, fidelity = model.optimize_theta_for_state(state)
        self.assertGreater(fidelity, 0.99999)

    def test_dark_state_hamiltonian_and_leading_order_vectors(self) -> None:
        config = Evered2023DarkStateConfig(
            omega_blue=1.2,
            omega_red=3.0,
            intermediate_detuning=20.0,
            two_photon_detuning=-0.4,
        )
        hamiltonian = np.asarray(config.hamiltonian().full(), dtype=np.complex128)
        self.assertEqual(hamiltonian.shape, (3, 3))
        self.assertAlmostEqual(float(hamiltonian[0, 1].real), 0.6)
        self.assertAlmostEqual(float(hamiltonian[1, 1].real), -20.0)
        self.assertAlmostEqual(float(hamiltonian[1, 2].real), 1.5)
        self.assertAlmostEqual(float(hamiltonian[2, 2].real), 0.4)

        vectors = config.dark_bright_eigenvectors_leading_order()
        dark = vectors["D"]
        self.assertAlmostEqual(float(np.linalg.norm(dark)), 1.0)
        self.assertAlmostEqual(abs(complex(dark[1])), 0.0)
        self.assertGreater(abs(complex(vectors["B"][1])), 0.0)

    def test_detuning_sign_uses_negative_phase_derivative(self) -> None:
        pulse = Evered2023TimeOptimalPulse(static_detuning=0.13)
        times = np.linspace(0.0, pulse.dimensionless_duration, 7)
        np.testing.assert_allclose(pulse.two_photon_detuning(times), -pulse.phase_derivative(times))

    def test_two_photon_ladder_builder_keeps_evered_control_shape(self) -> None:
        model = build_evered2023_two_photon_ladder_model(
            species=idealised_yb171(),
            lower_rabi=1.0,
            upper_rabi=5.0,
            intermediate_detuning=20.0,
            blockade_shift=50.0,
            two_photon_detuning=0.02,
        )
        self.assertEqual(model.dimension(), 9)
        self.assertEqual(len(model.phase_control_hamiltonians()), 1)
        self.assertEqual(model.phase_control_amplitudes(), (1.0,))

    def test_experimental_scale_records_reported_values(self) -> None:
        calibration = Evered2023ParallelCZCalibration()
        payload = calibration.to_json()
        self.assertEqual(payload["atom_species"], "87Rb")
        self.assertEqual(payload["rydberg_state_n"], 53)
        self.assertAlmostEqual(float(payload["omega_over_2pi_hz"]), 4.6e6)
        self.assertAlmostEqual(float(payload["blue_rabi_hz"]), 237.0e6)
        self.assertAlmostEqual(float(payload["red_rabi_hz"]), 303.0e6)
        self.assertAlmostEqual(float(payload["intermediate_detuning_hz"]), 7.8e9)
        self.assertAlmostEqual(float(payload["reported_parallel_cz_fidelity"]), 0.995)

    def test_paper_rabi_light_shift_resonance_calibration_scale(self) -> None:
        calibration = Evered2023ParallelCZCalibration()
        omega_b = calibration.blue_rabi_hz / calibration.omega_over_2pi_hz
        omega_r = calibration.red_rabi_hz / calibration.omega_over_2pi_hz
        detuning = calibration.intermediate_detuning_hz / calibration.omega_over_2pi_hz
        delta_res = (omega_r**2 - omega_b**2) / (4.0 * detuning)
        self.assertAlmostEqual(delta_res, 0.2483277591973245)
        self.assertAlmostEqual(delta_res * calibration.omega_over_2pi_hz, 1.1423076923076928e6)

    def test_parameterized_grape_gradient_matches_finite_difference(self) -> None:
        model = build_evered2023_ideal_global_cz_model(species=idealised_yb171())
        optimizer = ClosedSystemGRAPE.evered_parameterized(
            model=model,
            omega_t_over_2pi=1.18,
            config=Evered2023ParameterizedGRAPEConfig(
                num_tslots=8,
                max_iter=1,
                num_restarts=1,
                fix_static_detuning=True,
            ),
        )
        variables = np.array([0.7, 1.05, -0.6, 2.1], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        epsilon = 1e-6
        for index in range(variables.size):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)

    def test_asymmetric_two_photon_parameterized_grape_gradient_matches_finite_difference(self) -> None:
        model = build_evered2023_two_photon_ladder_model(
            species=idealised_yb171(),
            lower_rabi=2.1,
            upper_rabi=2.8,
            intermediate_detuning=7.0,
            blockade_shift=20.0,
        )
        optimizer = ClosedSystemGRAPE.evered_parameterized(
            model=model,
            omega_t_over_2pi=1.18,
            config=Evered2023ParameterizedGRAPEConfig(
                num_tslots=5,
                max_iter=1,
                num_restarts=1,
                fix_static_detuning=False,
            ),
        )
        variables = np.array([0.7, 1.05, -0.6, 0.08, 2.1], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        epsilon = 1e-6
        for index in range(variables.size):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)

    def test_two_photon_detuning_hamiltonian_terms(self) -> None:
        model = build_evered2023_two_photon_detuning_model(
            species=idealised_yb171(),
            intermediate_detuning_over_effective_rabi=20.0,
            blockade_shift_over_effective_rabi=50.0,
        )
        self.assertEqual(model.dimension(), 9)
        h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
        h_delta = np.asarray(model.detuning_control_hamiltonian().full(), dtype=np.complex128)
        self.assertAlmostEqual(float(h_d[1, 1].real), -20.0)
        self.assertAlmostEqual(float(h_d[5, 5].real), -40.0)
        self.assertAlmostEqual(float(h_d[8, 8].real), 50.0)
        self.assertAlmostEqual(float(h_delta[2, 2].real), -1.0)
        self.assertAlmostEqual(float(h_delta[6, 6].real), -1.0)
        self.assertAlmostEqual(float(h_delta[8, 8].real), -2.0)

    def test_two_photon_detuning_model_uses_dressed_branch_vectors(self) -> None:
        model = build_evered2023_two_photon_detuning_model(
            species=idealised_yb171(),
            intermediate_detuning_over_effective_rabi=20.0,
            blockade_shift_over_effective_rabi=50.0,
            blue_rabi_over_effective_rabi=2.0,
            red_rabi_over_effective_rabi=3.0,
            static_resonance_shift=0.07,
            use_leading_order_dressed_basis=True,
        )
        self.assertAlmostEqual(model.static_resonance_shift, 0.07)
        self.assertAlmostEqual(model.leading_order_blue_dressing_epsilon(), 0.05)
        branch_01, branch_11 = model.phase_gate_branch_projectors()
        self.assertAlmostEqual(float(np.linalg.norm(branch_01)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(branch_11)), 1.0)
        self.assertGreater(abs(complex(branch_01[1])), 0.0)
        self.assertGreater(abs(complex(branch_11[4])), 0.0)

    def test_two_photon_detuning_grape_gradient_matches_finite_difference(self) -> None:
        model = build_evered2023_two_photon_detuning_model(
            species=idealised_yb171(),
            intermediate_detuning_over_effective_rabi=8.0,
            blockade_shift_over_effective_rabi=20.0,
        )
        optimizer = ClosedSystemGRAPE.evered_detuning(
            model=model,
            omega_t_over_2pi=1.18,
            config=Evered2023ParameterizedGRAPEConfig(
                num_tslots=5,
                max_iter=1,
                num_restarts=1,
                fix_static_detuning=True,
            ),
        )
        variables = np.array([0.7, 1.05, -0.6, 2.1], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        epsilon = 1e-6
        for index in range(variables.size):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)

    def test_diagonal_average_gate_fidelity_matches_existing_cz_formula(self) -> None:
        theta = 0.37
        alpha = 0.91 * np.exp(1j * theta)
        beta = -0.82 * np.exp(2j * theta)
        phased_sum = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
        population_sum = 1.0 + 2.0 * abs(alpha) ** 2 + abs(beta) ** 2
        expected = (abs(phased_sum) ** 2 + population_sum) / 20.0
        self.assertAlmostEqual(diagonal_cz_average_gate_fidelity(alpha, beta, theta), expected)

    def test_evered_exponential_decay_fidelity_is_one_for_exact_cz(self) -> None:
        theta = 0.23
        alpha = np.exp(1j * theta)
        beta = -np.exp(2j * theta)
        counts = (0, 2, 4, 6, 8)
        fidelities = repeated_diagonal_cz_average_fidelities(alpha, beta, theta, counts)
        np.testing.assert_allclose(fidelities, np.ones(len(counts)), atol=1e-12)
        benchmark = evered2023_exponential_decay_fidelity_from_diagonal_map(alpha, beta, theta, counts)
        self.assertAlmostEqual(benchmark.gate_fidelity, 1.0, places=12)

    def test_evered_exponential_decay_fidelity_detects_lossy_gate(self) -> None:
        theta = -0.41
        alpha = 0.98 * np.exp(1j * theta)
        beta = -0.96 * np.exp(2j * theta)
        benchmark = evered2023_exponential_decay_fidelity_from_diagonal_map(
            alpha,
            beta,
            theta,
            gate_counts=(0, 2, 4, 6, 8, 10),
        )
        self.assertLess(benchmark.gate_fidelity, 1.0)
        self.assertGreater(benchmark.gate_fidelity, 0.0)


if __name__ == "__main__":
    unittest.main()
