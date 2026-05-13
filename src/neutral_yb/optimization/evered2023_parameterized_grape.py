from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from neutral_yb.models.evered2023_parallel_cz import Evered2023TimeOptimalPulse
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


@dataclass(frozen=True)
class Evered2023ParameterizedGRAPEConfig:
    num_tslots: int = 160
    max_iter: int = 260
    num_restarts: int = 20
    seed: int = 73
    fix_static_detuning: bool = True
    static_detuning_value: float = 0.0
    fidelity_target: float = 0.9999
    show_progress: bool = False


@dataclass(frozen=True)
class Evered2023ParameterizedGRAPEResult:
    omega_t_over_2pi: float
    amplitude_phase_modulation: float
    phase_rate: float
    phase_offset: float
    static_detuning: float
    theta: float
    fidelity: float
    objective: float
    iterations: int
    success: bool
    message: str
    num_tslots: int
    num_restarts: int
    wall_time: float

    @property
    def dimensionless_duration(self) -> float:
        return 2.0 * np.pi * self.omega_t_over_2pi

    def pulse(self) -> Evered2023TimeOptimalPulse:
        return Evered2023TimeOptimalPulse(
            amplitude_phase_modulation=self.amplitude_phase_modulation,
            phase_rate=self.phase_rate,
            phase_offset=self.phase_offset,
            static_detuning=self.static_detuning,
            omega_t_over_2pi=self.omega_t_over_2pi,
        )

    def parameter_errors_vs_paper(self) -> dict[str, float]:
        paper = Evered2023TimeOptimalPulse()
        return {
            "amplitude_phase_modulation": float(self.amplitude_phase_modulation - paper.amplitude_phase_modulation),
            "amplitude_phase_modulation_over_2pi": float(
                (self.amplitude_phase_modulation - paper.amplitude_phase_modulation) / (2.0 * np.pi)
            ),
            "phase_rate": float(self.phase_rate - paper.phase_rate),
            "phase_offset": float(self.phase_offset - paper.phase_offset),
            "static_detuning": float(self.static_detuning - paper.static_detuning),
            "omega_t_over_2pi": float(self.omega_t_over_2pi - paper.omega_t_over_2pi),
        }

    def to_json(self) -> dict[str, float | int | bool | str | dict[str, float]]:
        return {
            "omega_t_over_2pi": float(self.omega_t_over_2pi),
            "dimensionless_duration_omega_t": float(self.dimensionless_duration),
            "amplitude_phase_modulation_rad": float(self.amplitude_phase_modulation),
            "amplitude_phase_modulation_over_2pi": float(self.amplitude_phase_modulation / (2.0 * np.pi)),
            "phase_rate_over_omega": float(self.phase_rate),
            "phase_offset_rad": float(self.phase_offset),
            "static_detuning_over_omega": float(self.static_detuning),
            "theta": float(self.theta),
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "num_tslots": int(self.num_tslots),
            "num_restarts": int(self.num_restarts),
            "wall_time": float(self.wall_time),
            "parameter_errors_vs_paper": self.parameter_errors_vs_paper(),
        }


class Evered2023ParameterizedGRAPEOptimizer:
    """GRAPE optimizer for the Evered fixed-amplitude analytic pulse family.

    The optimized variables are the global pulse parameters in

        phi(t) = A cos(omega t - phi0) + delta0 t.

    Slot-wise GRAPE phase gradients are computed by `PaperGlobalPhaseOptimizer`
    and then chained to the global parameters. The paper profile is used only as
    a reference for reporting errors, not as an initial condition.
    """

    def __init__(
        self,
        *,
        model,
        omega_t_over_2pi: float,
        config: Evered2023ParameterizedGRAPEConfig,
    ) -> None:
        self.model = model
        self.omega_t_over_2pi = float(omega_t_over_2pi)
        self.evo_time = 2.0 * np.pi * self.omega_t_over_2pi
        self.config = config
        self.times = (
            np.arange(int(config.num_tslots), dtype=np.float64) + 0.5
        ) * (self.evo_time / int(config.num_tslots))
        self.slot_optimizer = PaperGlobalPhaseOptimizer(
            model=model,
            config=GlobalPhaseOptimizationConfig(
                num_tslots=int(config.num_tslots),
                evo_time=self.evo_time,
                max_iter=1,
            ),
        )

    def optimize(self) -> Evered2023ParameterizedGRAPEResult:
        rng = np.random.default_rng(self.config.seed)
        best: Evered2023ParameterizedGRAPEResult | None = None
        started_at = time.perf_counter()
        for restart in range(int(self.config.num_restarts)):
            variables0 = self._initial_variables(rng)
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=self._bounds(),
                options={"maxiter": int(self.config.max_iter), "ftol": 1e-12, "gtol": 1e-8},
            )
            candidate = self._result_from_variables(
                result.x,
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
                wall_time=time.perf_counter() - started_at,
            )
            if self.config.show_progress:
                print(
                    f"[evered2023-param-grape] T={self.omega_t_over_2pi:.4f} "
                    f"restart={restart + 1}/{self.config.num_restarts} "
                    f"F={candidate.fidelity:.10f} "
                    f"A/2pi={candidate.amplitude_phase_modulation / (2.0 * np.pi):.5f} "
                    f"omega={candidate.phase_rate:.5f} phi0={candidate.phase_offset:.5f}",
                    flush=True,
                )
            if best is None or candidate.fidelity > best.fidelity:
                best = candidate
        assert best is not None
        return best

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        full = self._unpack_variables(variables)
        amplitude, phase_rate, phase_offset, static_detuning, theta = full
        phases = self._phases(full)
        objective, slot_gradient = self.slot_optimizer.objective_and_gradient(
            np.concatenate([phases, np.array([theta], dtype=np.float64)])
        )
        phase_gradient = slot_gradient[:-1]
        theta_gradient = slot_gradient[-1]
        argument = phase_rate * self.times - phase_offset
        gradients = [
            float(np.sum(phase_gradient * np.cos(argument))),
            float(np.sum(phase_gradient * (-amplitude * self.times * np.sin(argument)))),
            float(np.sum(phase_gradient * (amplitude * np.sin(argument)))),
        ]
        if not self.config.fix_static_detuning:
            gradients.append(float(np.sum(phase_gradient * self.times)))
        gradients.append(float(theta_gradient))
        return float(objective), np.asarray(gradients, dtype=np.float64)

    def sampled_phases(self, result: Evered2023ParameterizedGRAPEResult) -> np.ndarray:
        return self._phases(
            np.array(
                [
                    result.amplitude_phase_modulation,
                    result.phase_rate,
                    result.phase_offset,
                    result.static_detuning,
                    result.theta,
                ],
                dtype=np.float64,
            )
        )

    def _phases(self, full_variables: np.ndarray) -> np.ndarray:
        amplitude, phase_rate, phase_offset, static_detuning, _theta = full_variables
        return amplitude * np.cos(phase_rate * self.times - phase_offset) + static_detuning * self.times

    def _unpack_variables(self, variables: np.ndarray) -> np.ndarray:
        variables = np.asarray(variables, dtype=np.float64)
        if self.config.fix_static_detuning:
            return np.array(
                [
                    variables[0],
                    variables[1],
                    variables[2],
                    float(self.config.static_detuning_value),
                    variables[3],
                ],
                dtype=np.float64,
            )
        return variables

    def _initial_variables(self, rng: np.random.Generator) -> np.ndarray:
        base = [
            float(rng.uniform(0.05, 1.6)),
            float(rng.uniform(0.4, 1.8)),
            float(rng.uniform(-np.pi, np.pi)),
        ]
        if not self.config.fix_static_detuning:
            base.append(float(rng.uniform(-0.3, 0.3)))
        base.append(float(rng.uniform(0.0, 2.0 * np.pi)))
        return np.asarray(base, dtype=np.float64)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [
            (0.0, 2.0 * np.pi),
            (0.2, 2.5),
            (-np.pi, np.pi),
        ]
        if not self.config.fix_static_detuning:
            bounds.append((-1.0, 1.0))
        bounds.append((0.0, 2.0 * np.pi))
        return bounds

    def _result_from_variables(
        self,
        variables: np.ndarray,
        *,
        iterations: int,
        success: bool,
        message: str,
        wall_time: float,
    ) -> Evered2023ParameterizedGRAPEResult:
        full = self._unpack_variables(variables)
        objective, _gradient = self.objective_and_gradient(variables)
        return Evered2023ParameterizedGRAPEResult(
            omega_t_over_2pi=self.omega_t_over_2pi,
            amplitude_phase_modulation=float(full[0]),
            phase_rate=float(full[1]),
            phase_offset=float(full[2]),
            static_detuning=float(full[3]),
            theta=float(np.mod(full[4], 2.0 * np.pi)),
            fidelity=float(1.0 - objective),
            objective=float(objective),
            iterations=int(iterations),
            success=bool(success),
            message=message,
            num_tslots=int(self.config.num_tslots),
            num_restarts=int(self.config.num_restarts),
            wall_time=float(wall_time),
        )


class Evered2023TwoPhotonDetuningGRAPEOptimizer:
    """Parameterized GRAPE for the two-photon detuning-gauge Hamiltonian.

    The physical control is the two-photon detuning

        delta(t) = -d phi(t) / dt
                 = A * omega * sin(omega * t - phi0) - delta0.

    This follows the Evered Methods convention where the 420-nm phase is
    represented by the time-dependent two-photon detuning in Eq. (2).
    """

    def __init__(
        self,
        *,
        model,
        omega_t_over_2pi: float,
        config: Evered2023ParameterizedGRAPEConfig,
    ) -> None:
        self.model = model
        self.omega_t_over_2pi = float(omega_t_over_2pi)
        self.evo_time = 2.0 * np.pi * self.omega_t_over_2pi
        self.config = config
        self.times = (
            np.arange(int(config.num_tslots), dtype=np.float64) + 0.5
        ) * (self.evo_time / int(config.num_tslots))
        self.dt = self.evo_time / int(config.num_tslots)
        self.h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
        self.h_delta = np.asarray(model.detuning_control_hamiltonian().full(), dtype=np.complex128)
        self.initial_state = np.asarray(model.initial_state().full(), dtype=np.complex128).ravel()
        self.dimension = int(self.initial_state.size)
        self.phase_gate_indices = tuple(int(index) for index in model.phase_gate_state_indices())

    def optimize(self) -> Evered2023ParameterizedGRAPEResult:
        rng = np.random.default_rng(self.config.seed)
        best: Evered2023ParameterizedGRAPEResult | None = None
        started_at = time.perf_counter()
        for restart in range(int(self.config.num_restarts)):
            variables0 = self._initial_variables(rng)
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=self._bounds(),
                options={"maxiter": int(self.config.max_iter), "ftol": 1e-12, "gtol": 1e-8},
            )
            candidate = self._result_from_variables(
                result.x,
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
                wall_time=time.perf_counter() - started_at,
            )
            if self.config.show_progress:
                print(
                    f"[evered2023-2photon-grape] T={self.omega_t_over_2pi:.4f} "
                    f"restart={restart + 1}/{self.config.num_restarts} "
                    f"F={candidate.fidelity:.10f} "
                    f"A/2pi={candidate.amplitude_phase_modulation / (2.0 * np.pi):.5f} "
                    f"omega={candidate.phase_rate:.5f} phi0={candidate.phase_offset:.5f} "
                    f"delta0={candidate.static_detuning:.5f}",
                    flush=True,
                )
            if best is None or candidate.fidelity > best.fidelity:
                best = candidate
        assert best is not None
        return best

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        full = self._unpack_variables(variables)
        amplitude, phase_rate, phase_offset, static_detuning, theta = full
        detunings = self._detunings(full)

        slice_unitaries: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_state]
        state = np.array(self.initial_state, copy=True)
        for detuning in detunings:
            h_k = self.h_d + float(detuning) * self.h_delta
            u_k = expm(-1j * self.dt * h_k)
            slice_unitaries.append(u_k)
            state = u_k @ state
            state_prefix.append(state)

        final_state = state_prefix[-1]
        alpha = final_state[self.phase_gate_indices[0]]
        beta = final_state[self.phase_gate_indices[1]]
        s_value = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
        fidelity = self.model.phase_gate_fidelity(final_state, theta)
        objective = 1.0 - fidelity

        suffix_unitaries: list[np.ndarray] = [
            np.eye(self.dimension, dtype=np.complex128) for _ in range(int(self.config.num_tslots))
        ]
        suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(int(self.config.num_tslots) - 1, -1, -1):
            suffix_unitaries[index] = suffix
            suffix = suffix @ slice_unitaries[index]

        detuning_gradient = np.zeros(int(self.config.num_tslots), dtype=np.float64)
        for index, detuning in enumerate(detunings):
            h_k = self.h_d + float(detuning) * self.h_delta
            du_k = expm_frechet(-1j * self.dt * h_k, -1j * self.dt * self.h_delta, compute_expm=False)
            d_state = suffix_unitaries[index] @ du_k @ state_prefix[index]
            d_alpha = d_state[self.phase_gate_indices[0]]
            d_beta = d_state[self.phase_gate_indices[1]]
            d_s = 2.0 * np.exp(-1j * theta) * d_alpha - np.exp(-2j * theta) * d_beta
            d_pop = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
            d_fidelity = (2.0 * np.real(np.conj(s_value) * d_s) + d_pop) / 20.0
            detuning_gradient[index] = -float(d_fidelity)

        d_s_theta = -2j * np.exp(-1j * theta) * alpha + 2j * np.exp(-2j * theta) * beta
        d_fidelity_theta = 2.0 * np.real(np.conj(s_value) * d_s_theta) / 20.0

        argument = phase_rate * self.times - phase_offset
        gradients = [
            float(np.sum(detuning_gradient * (phase_rate * np.sin(argument)))),
            float(np.sum(detuning_gradient * (amplitude * np.sin(argument) + amplitude * phase_rate * self.times * np.cos(argument)))),
            float(np.sum(detuning_gradient * (-amplitude * phase_rate * np.cos(argument)))),
        ]
        if not self.config.fix_static_detuning:
            gradients.append(float(np.sum(detuning_gradient * -1.0)))
        gradients.append(float(-d_fidelity_theta))
        return float(objective), np.asarray(gradients, dtype=np.float64)

    def final_state(self, result: Evered2023ParameterizedGRAPEResult) -> np.ndarray:
        state = np.array(self.initial_state, copy=True)
        for detuning in self.sampled_detunings(result):
            state = expm(-1j * self.dt * (self.h_d + float(detuning) * self.h_delta)) @ state
        return state

    def trajectory(self, result: Evered2023ParameterizedGRAPEResult) -> tuple[np.ndarray, list[np.ndarray]]:
        state = np.array(self.initial_state, copy=True)
        states = [state]
        for detuning in self.sampled_detunings(result):
            state = expm(-1j * self.dt * (self.h_d + float(detuning) * self.h_delta)) @ state
            states.append(state)
        return np.linspace(0.0, self.evo_time, int(self.config.num_tslots) + 1), states

    def sampled_detunings(self, result: Evered2023ParameterizedGRAPEResult) -> np.ndarray:
        return self._detunings(
            np.array(
                [
                    result.amplitude_phase_modulation,
                    result.phase_rate,
                    result.phase_offset,
                    result.static_detuning,
                    result.theta,
                ],
                dtype=np.float64,
            )
        )

    def sampled_phases(self, result: Evered2023ParameterizedGRAPEResult) -> np.ndarray:
        amplitude = result.amplitude_phase_modulation
        phase_rate = result.phase_rate
        phase_offset = result.phase_offset
        static_detuning = result.static_detuning
        return amplitude * np.cos(phase_rate * self.times - phase_offset) + static_detuning * self.times

    def _detunings(self, full_variables: np.ndarray) -> np.ndarray:
        amplitude, phase_rate, phase_offset, static_detuning, _theta = full_variables
        argument = phase_rate * self.times - phase_offset
        return amplitude * phase_rate * np.sin(argument) - static_detuning

    def _unpack_variables(self, variables: np.ndarray) -> np.ndarray:
        variables = np.asarray(variables, dtype=np.float64)
        if self.config.fix_static_detuning:
            return np.array(
                [
                    variables[0],
                    variables[1],
                    variables[2],
                    float(self.config.static_detuning_value),
                    variables[3],
                ],
                dtype=np.float64,
            )
        return variables

    def _initial_variables(self, rng: np.random.Generator) -> np.ndarray:
        base = [
            float(rng.uniform(0.05, 1.6)),
            float(rng.uniform(0.4, 1.8)),
            float(rng.uniform(-np.pi, np.pi)),
        ]
        if not self.config.fix_static_detuning:
            base.append(float(rng.uniform(-0.3, 0.3)))
        base.append(float(rng.uniform(0.0, 2.0 * np.pi)))
        return np.asarray(base, dtype=np.float64)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [
            (0.0, 2.0 * np.pi),
            (0.2, 2.5),
            (-np.pi, np.pi),
        ]
        if not self.config.fix_static_detuning:
            bounds.append((-1.0, 1.0))
        bounds.append((0.0, 2.0 * np.pi))
        return bounds

    def _result_from_variables(
        self,
        variables: np.ndarray,
        *,
        iterations: int,
        success: bool,
        message: str,
        wall_time: float,
    ) -> Evered2023ParameterizedGRAPEResult:
        full = self._unpack_variables(variables)
        objective, _gradient = self.objective_and_gradient(variables)
        return Evered2023ParameterizedGRAPEResult(
            omega_t_over_2pi=self.omega_t_over_2pi,
            amplitude_phase_modulation=float(full[0]),
            phase_rate=float(full[1]),
            phase_offset=float(full[2]),
            static_detuning=float(full[3]),
            theta=float(np.mod(full[4], 2.0 * np.pi)),
            fidelity=float(1.0 - objective),
            objective=float(objective),
            iterations=int(iterations),
            success=bool(success),
            message=message,
            num_tslots=int(self.config.num_tslots),
            num_restarts=int(self.config.num_restarts),
            wall_time=float(wall_time),
        )
