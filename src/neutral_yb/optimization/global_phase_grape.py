from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Protocol

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize


class PhaseGateModel(Protocol):
    species: object

    def drift_hamiltonian(self): ...
    def control_hamiltonians(self): ...
    def initial_state(self): ...
    def phase_gate_fidelity(self, state: np.ndarray, theta: float) -> float: ...


@dataclass(frozen=True)
class GlobalPhaseOptimizationConfig:
    num_tslots: int = 100
    evo_time: float = 8.8
    max_iter: int = 300
    phase_seed: int = 11
    init_phase_spread: float = 0.8
    fidelity_target: float = 0.999
    smoothness_weight: float = 0.0
    num_restarts: int = 1

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class GlobalPhaseOptimizationResult:
    phases: np.ndarray
    theta: float
    fidelity: float
    objective: float
    iterations: int
    success: bool
    message: str
    evo_time: float
    num_tslots: int
    smoothness_cost: float

    def to_json(self) -> dict[str, float | int | bool | str | list[float]]:
        return {
            "phases": [float(x) for x in self.phases],
            "theta": float(self.theta),
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "smoothness_cost": float(self.smoothness_cost),
        }


@dataclass(frozen=True)
class TimeOptimalScanResult:
    durations: list[float]
    fidelities: list[float]
    best_duration: float | None
    best_fidelity: float
    target_reached: bool

    def to_json(self) -> dict[str, float | bool | list[float] | None]:
        return {
            "durations": [float(x) for x in self.durations],
            "fidelities": [float(x) for x in self.fidelities],
            "best_duration": None if self.best_duration is None else float(self.best_duration),
            "best_fidelity": float(self.best_fidelity),
            "target_reached": bool(self.target_reached),
        }


class PaperGlobalPhaseOptimizer:
    """Paper-style optimizer for the ideal global CZ problem."""

    def __init__(self, model: PhaseGateModel, config: GlobalPhaseOptimizationConfig):
        self.model = model
        self.config = config
        self.h_d = model.drift_hamiltonian().full()
        self.h_x, self.h_y = [operator.full() for operator in model.control_hamiltonians()]
        self.initial_state = model.initial_state().full().ravel()
        self.dimension = self.initial_state.shape[0]
        self.omega_max = model.species.omega_max

    def initial_phases(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed)
        phases = rng.normal(0.0, self.config.init_phase_spread, size=self.config.num_tslots)
        return np.mod(phases, 2.0 * np.pi)

    def optimize(
        self,
        initial_phases: np.ndarray | None = None,
        initial_theta: float = 0.0,
    ) -> GlobalPhaseOptimizationResult:
        best_result: GlobalPhaseOptimizationResult | None = None
        base_phases = self.initial_phases() if initial_phases is None else np.asarray(initial_phases, dtype=np.float64)

        for restart in range(self.config.num_restarts):
            phases0 = base_phases if restart == 0 else self._jittered_initial_phases(base_phases, restart)
            variables0 = np.concatenate([np.asarray(phases0, dtype=np.float64), np.array([initial_theta])])
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, 2.0 * np.pi)] * len(variables0),
                options={"maxiter": self.config.max_iter},
            )

            phases = np.mod(result.x[:-1], 2.0 * np.pi)
            theta = float(np.mod(result.x[-1], 2.0 * np.pi))
            final_state = self.final_state(phases)
            fidelity = self.model.phase_gate_fidelity(final_state, theta)
            smoothness_cost = self._smoothness_cost(phases)

            candidate = GlobalPhaseOptimizationResult(
                phases=phases,
                theta=theta,
                fidelity=float(fidelity),
                objective=float(1.0 - fidelity + self.config.smoothness_weight * smoothness_cost),
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
                evo_time=float(self.config.evo_time),
                num_tslots=int(self.config.num_tslots),
                smoothness_cost=float(smoothness_cost),
            )
            if best_result is None or self._is_better(candidate, best_result):
                best_result = candidate

        assert best_result is not None
        return best_result

    def scan_durations(
        self,
        durations: list[float],
        initial_phases: np.ndarray | None = None,
    ) -> tuple[TimeOptimalScanResult, list[GlobalPhaseOptimizationResult]]:
        phases = self.initial_phases() if initial_phases is None else np.asarray(initial_phases, dtype=np.float64)
        theta = 0.0
        results: list[GlobalPhaseOptimizationResult] = []
        fidelities: list[float] = []

        for duration in durations:
            optimizer = PaperGlobalPhaseOptimizer(
                self.model,
                GlobalPhaseOptimizationConfig(
                    num_tslots=self.config.num_tslots,
                    evo_time=duration,
                    max_iter=self.config.max_iter,
                    phase_seed=self.config.phase_seed,
                    init_phase_spread=self.config.init_phase_spread,
                    fidelity_target=self.config.fidelity_target,
                    smoothness_weight=self.config.smoothness_weight,
                    num_restarts=self.config.num_restarts,
                ),
            )
            result = optimizer.optimize(phases, theta)
            results.append(result)
            fidelities.append(result.fidelity)
            phases = result.phases
            theta = result.theta

        qualified = [res for res in results if res.fidelity >= self.config.fidelity_target]
        best_duration = None if not qualified else min(res.evo_time for res in qualified)

        return (
            TimeOptimalScanResult(
                durations=list(durations),
                fidelities=fidelities,
                best_duration=best_duration,
                best_fidelity=max(fidelities) if fidelities else 0.0,
                target_reached=best_duration is not None,
            ),
            results,
        )

    def save_result(self, result: GlobalPhaseOptimizationResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def save_scan(self, result: TimeOptimalScanResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def final_state(self, phases: np.ndarray) -> np.ndarray:
        state = np.array(self.initial_state, copy=True)
        dt = self.config.dt
        for phase in phases:
            u_k = expm(-1j * dt * self._hamiltonian_from_phase(float(phase)))
            state = u_k @ state
        return state

    def trajectory(self, phases: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        state = np.array(self.initial_state, copy=True)
        states = [state]
        times = np.linspace(0.0, self.config.evo_time, len(phases) + 1)
        dt = self.config.dt
        for phase in phases:
            u_k = expm(-1j * dt * self._hamiltonian_from_phase(float(phase)))
            state = u_k @ state
            states.append(state)
        return times, states

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        phases = np.asarray(variables[:-1], dtype=np.float64)
        theta = float(variables[-1])
        dt = self.config.dt

        slice_unitaries: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_state]
        current_state = np.array(self.initial_state, copy=True)

        for phase in phases:
            h_k = self._hamiltonian_from_phase(float(phase))
            u_k = expm(-1j * dt * h_k)
            slice_unitaries.append(u_k)
            current_state = u_k @ current_state
            state_prefix.append(current_state)

        final_state = state_prefix[-1]
        alpha = final_state[0]
        beta = final_state[2]
        s = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta

        fidelity = self.model.phase_gate_fidelity(final_state, theta)
        smoothness_cost = self._smoothness_cost(phases)
        objective = 1.0 - fidelity + self.config.smoothness_weight * smoothness_cost

        suffix_unitaries: list[np.ndarray] = [np.eye(self.dimension, dtype=np.complex128)] * len(phases)
        current_suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(len(phases) - 1, -1, -1):
            suffix_unitaries[index] = current_suffix
            current_suffix = current_suffix @ slice_unitaries[index]

        gradient = np.zeros_like(variables)

        for index, phase in enumerate(phases):
            h_k = self._hamiltonian_from_phase(float(phase))
            dh_k = self._phase_derivative_hamiltonian(float(phase))
            du_k = expm_frechet(-1j * dt * h_k, -1j * dt * dh_k, compute_expm=False)
            d_state = suffix_unitaries[index] @ du_k @ state_prefix[index]

            d_alpha = d_state[0]
            d_beta = d_state[2]
            d_s = 2.0 * np.exp(-1j * theta) * d_alpha - np.exp(-2j * theta) * d_beta
            d_pop = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
            d_fidelity = (2.0 * np.real(np.conj(s) * d_s) + d_pop) / 20.0
            gradient[index] = -d_fidelity

        d_s_theta = -2j * np.exp(-1j * theta) * alpha + 2j * np.exp(-2j * theta) * beta
        d_fidelity_theta = 2.0 * np.real(np.conj(s) * d_s_theta) / 20.0
        gradient[-1] = -d_fidelity_theta

        if self.config.smoothness_weight > 0.0:
            gradient[:-1] += self.config.smoothness_weight * self._smoothness_gradient(phases)

        return float(objective), gradient

    def _hamiltonian_from_phase(self, phase: float) -> np.ndarray:
        return self.h_d + self.omega_max * (
            np.cos(phase) * self.h_x + np.sin(phase) * self.h_y
        )

    def _phase_derivative_hamiltonian(self, phase: float) -> np.ndarray:
        return self.omega_max * (-np.sin(phase) * self.h_x + np.cos(phase) * self.h_y)

    def _jittered_initial_phases(self, base_phases: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed + 1000 * restart)
        jitter = rng.normal(0.0, self.config.init_phase_spread * 0.35, size=base_phases.shape[0])
        return np.mod(base_phases + jitter, 2.0 * np.pi)

    @staticmethod
    def _is_better(left: GlobalPhaseOptimizationResult, right: GlobalPhaseOptimizationResult) -> bool:
        if left.fidelity > right.fidelity + 1e-8:
            return True
        if abs(left.fidelity - right.fidelity) <= 1e-8 and left.smoothness_cost < right.smoothness_cost:
            return True
        return False

    @staticmethod
    def _smoothness_cost(phases: np.ndarray) -> float:
        if len(phases) < 2:
            return 0.0
        delta = phases[1:] - phases[:-1]
        return float(np.mean(1.0 - np.cos(delta)))

    @staticmethod
    def _smoothness_gradient(phases: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(phases)
        if len(phases) < 2:
            return grad
        scale = 1.0 / (len(phases) - 1)
        delta = phases[1:] - phases[:-1]
        grad[0] = -np.sin(delta[0]) * scale
        grad[-1] = np.sin(delta[-1]) * scale
        if len(phases) > 2:
            grad[1:-1] = (np.sin(delta[:-1]) - np.sin(delta[1:])) * scale
        return grad
