from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from tests import _bootstrap  # noqa: F401

from neutral_yb.config.ma2023_calibration import (
    build_ma2023_model,
    build_ma2023_quasistatic_ensemble,
    ma2023_experimental_calibration,
    ma2023_fig3_controls,
)
from neutral_yb.models.ma2023_pulse import (
    Ma2023GaussianEdgePulse,
    controls_from_envelope_phase,
    validate_phase_only_pulse,
    wrap_phase,
)
from neutral_yb.models.ma2023_noise import (
    Ma2023NoiseTraceConfig,
    doppler_detuning_rms_from_t2_star,
    generate_noise_trace,
)
from neutral_yb.models.ma2023_six_level import Ma2023PerfectBlockadeSixLevelModel, Ma2023SixLevelNoiseConfig
from neutral_yb.models.ma2023_time_optimal_2q import Ma2023NoiseConfig
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer
from neutral_yb.optimization.ma2023_six_level_grape import (
    Ma2023SixLevelChebyshevPhaseRateOptimizer,
    Ma2023SixLevelGRAPEConfig,
    Ma2023SixLevelPhaseOptimizer,
)
from experiments.evaluate_ma2023_six_level_noisy import noisy_channel_summary


class Ma2023TimeOptimal2QModelTest(unittest.TestCase):
    def test_model_shape_and_collapse_ops(self) -> None:
        model = build_ma2023_model(include_noise=True)
        self.assertEqual(model.dimension(), 7)
        self.assertEqual(model.active_gate_indices(), (0, 2))
        self.assertEqual(model.drift_liouvillian().shape, (49, 49))
        self.assertGreater(len(model.collapse_operators()), 0)

    def test_noiseless_model_has_no_collapse_ops(self) -> None:
        model = build_ma2023_model(include_noise=False)
        self.assertEqual(model.noise, Ma2023NoiseConfig())
        self.assertEqual(model.collapse_operators(), [])

    def test_calibration_time_conversion_round_trip(self) -> None:
        calibration = ma2023_experimental_calibration()
        duration = calibration.target_dimensionless_duration
        seconds = calibration.dimensionless_gate_time_to_seconds(duration)
        self.assertAlmostEqual(calibration.physical_gate_time_to_dimensionless(seconds), duration)

    def test_imported_fig3_controls_load_when_processed_data_exists(self) -> None:
        ctrl_x, ctrl_y, duration = ma2023_fig3_controls(num_tslots=8)
        self.assertEqual(ctrl_x.shape, (8,))
        self.assertEqual(ctrl_y.shape, (8,))
        self.assertGreater(duration, 0.0)
        self.assertLessEqual(float(np.max(np.sqrt(ctrl_x**2 + ctrl_y**2))), 1.0)

    def test_phase_only_gaussian_edge_pulse_constraints(self) -> None:
        pulse = Ma2023GaussianEdgePulse(num_tslots=32)
        envelope = pulse.envelope()
        phases = wrap_phase(np.linspace(-4.0 * np.pi, 4.0 * np.pi, envelope.size))
        ctrl_x, ctrl_y = controls_from_envelope_phase(envelope, phases)
        amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2)
        validation = validate_phase_only_pulse(amplitudes, phases)
        self.assertTrue(validation["amplitude_starts_at_zero"])
        self.assertTrue(validation["amplitude_ends_at_zero"])
        self.assertTrue(validation["amplitude_within_bound"])
        self.assertTrue(validation["phase_within_minus_pi_pi"])
        self.assertAlmostEqual(float(envelope[0]), 0.0)
        self.assertAlmostEqual(float(envelope[-1]), 0.0)
        self.assertLessEqual(float(np.max(envelope)), 1.0)

    def test_methods_six_level_blockade_hamiltonian_terms(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(delta_r=5.8, include_loss_state=False)
        self.assertEqual(model.dimension(), 15)
        self.assertEqual(model.computational_indices(), (0, 5, 10))
        h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
        h_x, h_y = [
            np.asarray(operator.full(), dtype=np.complex128)
            for operator in model.lower_leg_control_hamiltonians()
        ]
        self.assertAlmostEqual(float(h_d[1, 1].real), -3.0 * 5.8)
        self.assertAlmostEqual(float(h_d[2, 2].real), -1.0 * 5.8)
        self.assertAlmostEqual(float(h_d[6, 6].real), -2.0 * 5.8)
        self.assertAlmostEqual(float(h_d[7, 7].real), 0.0)
        self.assertAlmostEqual(float(h_x[0, 1].real), 0.5)
        self.assertAlmostEqual(float(h_x[0, 2].real), 0.5 / np.sqrt(3.0))
        self.assertAlmostEqual(float(h_x[5, 6].real), 0.5 / np.sqrt(3.0))
        self.assertAlmostEqual(float(h_x[5, 7].real), 0.5)
        self.assertAlmostEqual(complex(h_y[0, 1]), -0.5j)
        self.assertEqual(model.collapse_operators(), [])

    def test_methods_six_level_rydberg_decay_splits_detected_and_undetected_sinks(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(
            delta_r=5.8,
            include_loss_state=True,
            noise=Ma2023SixLevelNoiseConfig(
                rydberg_decay_rate=0.2,
                rydberg_decay_detected_fraction=0.25,
            ),
        )
        self.assertEqual(model.dimension(), 17)
        self.assertEqual(model.erasure_index(), 15)
        self.assertEqual(model.detected_decay_index(), 15)
        self.assertEqual(model.undetected_decay_index(), 16)
        collapse_ops = model.collapse_operators()
        self.assertEqual(len(collapse_ops), 24)

    def test_methods_six_level_grape_gradient_matches_finite_difference(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(delta_r=5.8, include_loss_state=False)
        envelope = Ma2023GaussianEdgePulse(num_tslots=5).envelope()
        optimizer = Ma2023SixLevelPhaseOptimizer(
            model=model,
            config=Ma2023SixLevelGRAPEConfig(num_tslots=5, evo_time=2.0, max_iter=1),
            envelope=envelope,
        )
        variables = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.0, 0.4], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        epsilon = 1e-6
        for index in (1, 3, 6):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)

    def test_methods_chebyshev_phase_rate_gradient_matches_finite_difference(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(delta_r=5.8, include_loss_state=False)
        envelope = Ma2023GaussianEdgePulse(num_tslots=6).envelope()
        optimizer = Ma2023SixLevelChebyshevPhaseRateOptimizer(
            model=model,
            config=Ma2023SixLevelGRAPEConfig(
                num_tslots=6,
                evo_time=2.0,
                max_iter=1,
                chebyshev_degree=3,
                phase_smoothness_weight=1e-4,
                phase_curvature_weight=1e-4,
                optimize_phase_origin=True,
            ),
            envelope=envelope,
        )
        variables = np.array([0.2, -0.1, 0.05, -0.03, 0.4, 0.0, 0.3], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        epsilon = 1e-6
        for index in (0, 2, 4, 6):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)

    def test_methods_chebyshev_phase_rate_fixed_origin_gradient_matches_finite_difference(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(delta_r=5.8, include_loss_state=False)
        envelope = Ma2023GaussianEdgePulse(num_tslots=6).envelope()
        optimizer = Ma2023SixLevelChebyshevPhaseRateOptimizer(
            model=model,
            config=Ma2023SixLevelGRAPEConfig(
                num_tslots=6,
                evo_time=2.0,
                max_iter=1,
                chebyshev_degree=3,
                phase_smoothness_weight=1e-4,
                phase_curvature_weight=1e-4,
                optimize_phase_origin=False,
            ),
            envelope=envelope,
        )
        variables = np.array([0.2, -0.1, 0.05, -0.03, 0.0, 0.3], dtype=np.float64)
        _objective, gradient = optimizer.objective_and_gradient(variables)
        coefficients, phase_origin, _theta0, _theta1 = optimizer._unpack_chebyshev_variables(variables)
        phases = optimizer.phases_from_coefficients(coefficients, phase_origin)
        self.assertAlmostEqual(float(phases[0]), 0.0)
        epsilon = 1e-6
        for index in (0, 2, 5):
            plus = variables.copy()
            minus = variables.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            objective_plus, _ = optimizer.objective_and_gradient(plus)
            objective_minus, _ = optimizer.objective_and_gradient(minus)
            finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(float(gradient[index]), float(finite_difference), places=6)
        self.assertEqual(phase_origin, 0.0)

    def test_methods_noise_trace_and_lindblad_evolution_run(self) -> None:
        model = Ma2023PerfectBlockadeSixLevelModel(
            delta_r=5.8,
            include_loss_state=True,
            noise=Ma2023SixLevelNoiseConfig(rydberg_decay_rate=0.01),
        )
        envelope = Ma2023GaussianEdgePulse(num_tslots=4).envelope()
        optimizer = Ma2023SixLevelPhaseOptimizer(
            model=model,
            config=Ma2023SixLevelGRAPEConfig(num_tslots=4, evo_time=1.0, max_iter=1),
            envelope=envelope,
        )
        trace = generate_noise_trace(
            Ma2023NoiseTraceConfig(
                num_tslots=4,
                quasistatic_detuning_rms=0.01,
                intensity_noise_rms_fractional=0.01,
                phase_noise_rms_rad=0.01,
                seed=3,
            )
        )
        rho0 = np.zeros((model.dimension(), model.dimension()), dtype=np.complex128)
        rho0[model.computational_indices()[0], model.computational_indices()[0]] = 1.0
        rho_t = optimizer.evolve_density_matrix(envelope, np.zeros_like(envelope), rho0, noise_trace=trace)
        self.assertAlmostEqual(float(np.trace(rho_t).real), 1.0, places=9)
        sigma = doppler_detuning_rms_from_t2_star(t2_star_s=5.7e-6, omega_ref_rad_s=2.0 * np.pi * 1.59e6)
        self.assertGreater(sigma, 0.0)

    def test_lindblad_scoring_preserves_diagonal_gate_phase(self) -> None:
        class FakeModel:
            def dimension(self) -> int:
                return 3

            def computational_indices(self) -> tuple[int, int, int]:
                return 0, 1, 2

            def erasure_index(self):
                return None

            def undetected_decay_index(self):
                return None

        class FakeTrace:
            def validate(self, _num_tslots: int) -> None:
                return None

        class FakeOptimizer:
            model = FakeModel()

            def __init__(self, phases: np.ndarray) -> None:
                self.unitary = np.diag(phases)

            def evolve_density_matrix(self, _ctrl_x, _ctrl_y, rho0, *, noise_trace=None):
                return self.unitary @ rho0 @ self.unitary.conj().T

        theta0 = 0.3
        theta1 = -0.2
        target = np.array(
            [
                np.exp(1j * theta0),
                np.exp(1j * (theta0 + theta1)),
                -np.exp(1j * (theta0 + 2.0 * theta1)),
            ],
            dtype=np.complex128,
        )
        summary = noisy_channel_summary(
            FakeOptimizer(target),
            np.zeros(1),
            np.zeros(1),
            theta0,
            theta1,
            [FakeTrace()],
        )
        self.assertAlmostEqual(summary["process_fidelity"], 1.0)
        self.assertAlmostEqual(summary["gate_fidelity"], 1.0)

    def test_open_system_optimizer_objective_runs(self) -> None:
        model = build_ma2023_model(include_noise=True)
        ensemble = build_ma2023_quasistatic_ensemble(ensemble_size=2, seed=3, include_noise=True)
        optimizer = OpenSystemGRAPEOptimizer(
            model=model,
            config=OpenSystemGRAPEConfig(
                num_tslots=4,
                evo_time=2.0,
                max_iter=1,
                num_restarts=1,
                benchmark_active_channel=True,
            ),
            ensemble_models=ensemble,
        )
        ctrl_x, ctrl_y = optimizer.initial_guess()
        variables = np.concatenate([ctrl_x, ctrl_y, np.array([0.0])])
        objective, gradient = optimizer.objective_and_gradient(variables)
        self.assertGreaterEqual(objective, 0.0)
        self.assertEqual(gradient.shape, (9,))


if __name__ == "__main__":
    unittest.main()
