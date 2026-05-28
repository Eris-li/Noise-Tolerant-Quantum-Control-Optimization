from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import OptimizeResult, minimize

from neutral_yb.models.ma2023_pulse import gaussian_edge_envelope_from_times


def wrap_phase(phases: np.ndarray) -> np.ndarray:
    return (np.asarray(phases, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_for_plot(phases: list[float] | np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(phases, dtype=np.float64))


def resample_phase_controls(phases: list[float] | np.ndarray, target_slots: int) -> np.ndarray:
    old_grid = np.linspace(0.0, 1.0, len(phases), endpoint=True, dtype=np.float64)
    new_grid = np.linspace(0.0, 1.0, int(target_slots), endpoint=True, dtype=np.float64)
    return wrap_phase(np.interp(new_grid, old_grid, unwrap_for_plot(phases)))


def phase_regularization(
    phases: np.ndarray,
    smoothness_weight: float,
    curvature_weight: float,
) -> tuple[float, np.ndarray]:
    phases = np.asarray(phases, dtype=np.float64)
    cost = 0.0
    gradient = np.zeros_like(phases)
    if phases.size >= 2 and smoothness_weight > 0.0:
        delta = wrap_phase(phases[1:] - phases[:-1])
        cost += smoothness_weight * float(np.sum(delta**2))
        gradient[:-1] -= 2.0 * smoothness_weight * delta
        gradient[1:] += 2.0 * smoothness_weight * delta
    if phases.size >= 3 and curvature_weight > 0.0:
        delta = wrap_phase(phases[1:] - phases[:-1])
        curvature = wrap_phase(delta[1:] - delta[:-1])
        cost += curvature_weight * float(np.sum(curvature**2))
        gradient[:-2] += 2.0 * curvature_weight * curvature
        gradient[1:-1] -= 4.0 * curvature_weight * curvature
        gradient[2:] += 2.0 * curvature_weight * curvature
    return cost, gradient


@dataclass(frozen=True)
class ShelvedCRPhaseGRAPEConfig:
    omega_max_mhz: float
    total_time_ns: float
    edge_time_ns: float
    num_tslots: int = 64
    blockade_shift_mhz: float = 160.0
    smoothness_weight: float = 1e-4
    curvature_weight: float = 1e-4
    rydberg_lifetime_s: float | None = None


class ClosedShelvedCRPhaseGRAPE:
    """Phase-only GRAPE for the shelved control-Rydberg CZ segment.

    The reduced basis is ``|00>, |0c>, |0r>, |cc>, |W_cr>, |rr>``. Only the
    three diagonal entries ``|00>``, ``|0c>/<c0>`` and ``|cc>`` enter the
    reduced CZ scoring function; the weights expand them to the full 4D
    computational diagonal.
    """

    computational_indices = np.array([0, 1, 3], dtype=np.int64)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    alpha_coefficients = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    beta_coefficients = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    def __init__(self, config: ShelvedCRPhaseGRAPEConfig) -> None:
        self.config = config
        self.omega_max_mhz = float(config.omega_max_mhz)
        self.total_time_ns = float(config.total_time_ns)
        self.edge_time_ns = float(config.edge_time_ns)
        self.num_tslots = int(config.num_tslots)
        self.blockade_shift_mhz = float(config.blockade_shift_mhz)
        self.smoothness_weight = float(config.smoothness_weight)
        self.curvature_weight = float(config.curvature_weight)

        self.evo_time = 2.0 * np.pi * self.omega_max_mhz * 1e6 * self.total_time_ns * 1e-9
        self.dt = self.evo_time / self.num_tslots

        self.h_d = np.zeros((6, 6), dtype=np.complex128)
        self.h_d[5, 5] = self.blockade_shift_mhz / self.omega_max_mhz
        self.h_x = np.zeros_like(self.h_d)
        self.h_y = np.zeros_like(self.h_d)
        self._add_quadrature_coupling(1, 2, 0.5)
        self._add_quadrature_coupling(3, 4, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(4, 5, 1.0 / np.sqrt(2.0))
        self.g_d = -1j * self.h_d
        self.g_x = -1j * self.h_x
        self.g_y = -1j * self.h_y
        self.envelope = gaussian_edge_envelope_from_times(
            self.num_tslots,
            self.total_time_ns,
            self.edge_time_ns,
            sigma_to_edge=1.0 / 3.0,
        )

    def initial_guess(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grid = np.linspace(0.0, np.pi, self.num_tslots, endpoint=False, dtype=np.float64)
        phases = 0.6 * np.sin(2.0 * grid + rng.uniform(0.0, 2.0 * np.pi))
        phases += 0.2 * rng.normal(size=self.num_tslots)
        return np.concatenate([wrap_phase(phases), np.array([0.0, 0.0], dtype=np.float64)])

    def target_phases(self, alpha: float, beta: float) -> np.ndarray:
        return np.array(
            [
                np.exp(1j * alpha),
                np.exp(1j * (alpha + beta)),
                -np.exp(1j * (alpha + 2.0 * beta)),
            ],
            dtype=np.complex128,
        )

    def generator(self, amplitude: float, phase: float) -> np.ndarray:
        return self.g_d + float(amplitude) * (
            math.cos(float(phase)) * self.g_x + math.sin(float(phase)) * self.g_y
        )

    def optimize(self, starts: list[np.ndarray], max_iter: int) -> tuple[OptimizeResult, dict[str, object]]:
        bounds = [(-np.pi, np.pi)] * self.num_tslots + [(-np.pi, np.pi), (-np.pi, np.pi)]
        best_result: OptimizeResult | None = None
        best_eval: dict[str, object] | None = None
        for start in starts:
            result = minimize(
                self.objective_and_gradient,
                np.asarray(start, dtype=np.float64),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(max_iter), "ftol": 1e-11, "maxls": 30},
            )
            evaluated = self.evaluate(result.x)
            if best_eval is None or float(evaluated["fidelity"]) > float(best_eval["fidelity"]):
                best_result = result
                best_eval = evaluated
        if best_result is None or best_eval is None:
            raise RuntimeError("No GRAPE starts were supplied")
        return best_result, best_eval

    def evaluate(self, variables: np.ndarray) -> dict[str, object]:
        phases = wrap_phase(variables[: self.num_tslots])
        alpha = float(variables[-2])
        beta = float(variables[-1])
        unitary = self.propagate(phases)
        z_diag = np.diag(unitary)[self.computational_indices]
        fidelity, process_fidelity, active_population = self._fidelity_from_z(z_diag, alpha, beta)
        phase_argument = np.array([np.angle(z_diag[2]) - 2.0 * np.angle(z_diag[1]) + np.angle(z_diag[0]) - np.pi])
        return {
            "fidelity": fidelity,
            "process_fidelity": process_fidelity,
            "active_population": active_population,
            "leakage": float(1.0 - active_population),
            "cz_phase_error_rad": float(wrap_phase(phase_argument)[0]),
            "z_diag": [complex(value) for value in z_diag],
            "phases": phases,
            "alpha": alpha,
            "beta": beta,
            "envelope": self.envelope,
        }

    def propagate(self, phases: np.ndarray) -> np.ndarray:
        unitary = np.eye(6, dtype=np.complex128)
        for amplitude, phase in zip(self.envelope, phases):
            unitary = expm(self.dt * self.generator(float(amplitude), float(phase))) @ unitary
        return unitary

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        phases = wrap_phase(variables[: self.num_tslots])
        alpha = float(variables[-2])
        beta = float(variables[-1])
        propagators: list[np.ndarray] = []
        prefixes: list[np.ndarray] = [np.eye(6, dtype=np.complex128)]
        current = prefixes[0]
        for amplitude, phase in zip(self.envelope, phases):
            generator = self.generator(float(amplitude), float(phase))
            propagator = expm(self.dt * generator)
            propagators.append(propagator)
            current = propagator @ current
            prefixes.append(current)

        z_diag = np.diag(current)[self.computational_indices]
        target = self.target_phases(alpha, beta)
        conjugate_target = np.conj(target)
        overlap = np.sum(self.weights * conjugate_target * z_diag)
        process_fidelity = abs(overlap) ** 2 / 16.0
        active_population = np.sum(self.weights * np.abs(z_diag) ** 2) / 4.0
        fidelity = (4.0 * process_fidelity + active_population) / 5.0

        gradient = np.zeros_like(variables)
        suffixes: list[np.ndarray] = [np.eye(6, dtype=np.complex128) for _ in range(self.num_tslots)]
        current_suffix = np.eye(6, dtype=np.complex128)
        for index in range(self.num_tslots - 1, -1, -1):
            suffixes[index] = current_suffix
            current_suffix = current_suffix @ propagators[index]

        for index, (amplitude, phase) in enumerate(zip(self.envelope, phases)):
            generator = self.generator(float(amplitude), float(phase))
            d_generator = float(amplitude) * (
                -math.sin(float(phase)) * self.g_x + math.cos(float(phase)) * self.g_y
            )
            d_propagator = expm_frechet(self.dt * generator, self.dt * d_generator, compute_expm=False)
            d_unitary = suffixes[index] @ d_propagator @ prefixes[index]
            dz_diag = np.diag(d_unitary)[self.computational_indices]
            d_overlap = np.sum(self.weights * conjugate_target * dz_diag)
            d_process = 2.0 * np.real(np.conj(overlap) * d_overlap) / 16.0
            d_population = np.sum(self.weights * 2.0 * np.real(np.conj(z_diag) * dz_diag)) / 4.0
            gradient[index] = -(4.0 * d_process + d_population) / 5.0

        for offset, coefficients in enumerate([self.alpha_coefficients, self.beta_coefficients]):
            d_conjugate_target = -1j * coefficients * conjugate_target
            d_overlap = np.sum(self.weights * d_conjugate_target * z_diag)
            d_process = 2.0 * np.real(np.conj(overlap) * d_overlap) / 16.0
            gradient[-2 + offset] = -(4.0 * d_process) / 5.0

        regularization, regularization_gradient = phase_regularization(
            phases,
            self.smoothness_weight,
            self.curvature_weight,
        )
        gradient[: self.num_tslots] += regularization_gradient
        return float(1.0 - fidelity + regularization), gradient

    def _fidelity_from_z(self, z_diag: np.ndarray, alpha: float, beta: float) -> tuple[float, float, float]:
        target = self.target_phases(alpha, beta)
        overlap = np.sum(self.weights * np.conj(target) * z_diag)
        process_fidelity = float(abs(overlap) ** 2 / 16.0)
        active_population = float(np.sum(self.weights * np.abs(z_diag) ** 2) / 4.0)
        fidelity = float((4.0 * process_fidelity + active_population) / 5.0)
        return fidelity, process_fidelity, active_population

    def _add_quadrature_coupling(self, left: int, right: int, strength: float) -> None:
        self.h_x[left, right] = strength
        self.h_x[right, left] = strength
        self.h_y[left, right] = -1j * strength
        self.h_y[right, left] = 1j * strength


class RydbergDecayShelvedCRPhaseGRAPE(ClosedShelvedCRPhaseGRAPE):
    """Shelved CR phase GRAPE with Rydberg decay as a no-jump non-Hermitian term."""

    def __init__(self, config: ShelvedCRPhaseGRAPEConfig) -> None:
        if config.rydberg_lifetime_s is None:
            raise ValueError("RydbergDecayShelvedCRPhaseGRAPE requires rydberg_lifetime_s")
        super().__init__(config)
        self.rydberg_lifetime_s = float(config.rydberg_lifetime_s)
        self.decay_rate_over_omega = 1.0 / (2.0 * np.pi * self.omega_max_mhz * 1e6 * self.rydberg_lifetime_s)
        self.g_decay = -0.5 * np.diag(
            [0.0, 0.0, self.decay_rate_over_omega, 0.0, self.decay_rate_over_omega, 2.0 * self.decay_rate_over_omega]
        ).astype(np.complex128)

    def generator(self, amplitude: float, phase: float) -> np.ndarray:
        return super().generator(amplitude, phase) + self.g_decay

    def evaluate(self, variables: np.ndarray) -> dict[str, object]:
        phases = wrap_phase(variables[: self.num_tslots])
        alpha = float(variables[-2])
        beta = float(variables[-1])
        unitary = self.propagate(phases)
        z_diag = np.diag(unitary)[self.computational_indices]
        process_fidelity, active_population, no_jump_average_fidelity = self._no_jump_process_metrics_from_z(
            z_diag,
            alpha,
            beta,
        )
        return {
            "fidelity": process_fidelity,
            "process_fidelity": process_fidelity,
            "no_jump_average_fidelity": no_jump_average_fidelity,
            "active_population": active_population,
            "loss_proxy": float(1.0 - active_population),
            "z_diag": [complex(value) for value in z_diag],
            "phases": phases,
            "alpha": alpha,
            "beta": beta,
            "envelope": self.envelope,
        }

    def _no_jump_process_metrics_from_z(
        self,
        z_diag: np.ndarray,
        alpha: float,
        beta: float,
    ) -> tuple[float, float, float]:
        target = self.target_phases(alpha, beta)
        overlap = np.sum(self.weights * np.conj(target) * z_diag)
        process_fidelity = float(abs(overlap) ** 2 / 16.0)
        active_population = float(np.sum(self.weights * np.abs(z_diag) ** 2) / 4.0)
        no_jump_average_fidelity = float((4.0 * process_fidelity + active_population) / 5.0)
        return process_fidelity, active_population, no_jump_average_fidelity

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        phases = wrap_phase(variables[: self.num_tslots])
        alpha = float(variables[-2])
        beta = float(variables[-1])
        propagators: list[np.ndarray] = []
        prefixes: list[np.ndarray] = [np.eye(6, dtype=np.complex128)]
        current = prefixes[0]
        for amplitude, phase in zip(self.envelope, phases):
            generator = self.generator(float(amplitude), float(phase))
            propagator = expm(self.dt * generator)
            propagators.append(propagator)
            current = propagator @ current
            prefixes.append(current)

        z_diag = np.diag(current)[self.computational_indices]
        target = self.target_phases(alpha, beta)
        conjugate_target = np.conj(target)
        overlap = np.sum(self.weights * conjugate_target * z_diag)
        process_fidelity = abs(overlap) ** 2 / 16.0

        gradient = np.zeros_like(variables)
        suffixes: list[np.ndarray] = [np.eye(6, dtype=np.complex128) for _ in range(self.num_tslots)]
        current_suffix = np.eye(6, dtype=np.complex128)
        for index in range(self.num_tslots - 1, -1, -1):
            suffixes[index] = current_suffix
            current_suffix = current_suffix @ propagators[index]

        for index, (amplitude, phase) in enumerate(zip(self.envelope, phases)):
            generator = self.generator(float(amplitude), float(phase))
            d_generator = float(amplitude) * (
                -math.sin(float(phase)) * self.g_x + math.cos(float(phase)) * self.g_y
            )
            d_propagator = expm_frechet(self.dt * generator, self.dt * d_generator, compute_expm=False)
            d_unitary = suffixes[index] @ d_propagator @ prefixes[index]
            dz_diag = np.diag(d_unitary)[self.computational_indices]
            d_overlap = np.sum(self.weights * conjugate_target * dz_diag)
            d_process = 2.0 * np.real(np.conj(overlap) * d_overlap) / 16.0
            gradient[index] = -d_process

        for offset, coefficients in enumerate([self.alpha_coefficients, self.beta_coefficients]):
            d_conjugate_target = -1j * coefficients * conjugate_target
            d_overlap = np.sum(self.weights * d_conjugate_target * z_diag)
            d_process = 2.0 * np.real(np.conj(overlap) * d_overlap) / 16.0
            gradient[-2 + offset] = -d_process

        regularization, regularization_gradient = phase_regularization(
            phases,
            self.smoothness_weight,
            self.curvature_weight,
        )
        gradient[: self.num_tslots] += regularization_gradient
        return float(1.0 - process_fidelity + regularization), gradient
