from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Protocol

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize


class LinearPhaseGateModel(Protocol):
    def drift_hamiltonian(self): ...
    def control_hamiltonians(self): ...
    def initial_state(self): ...
    def phase_gate_fidelity(self, state: np.ndarray, theta: float) -> float: ...


@dataclass(frozen=True)
class LinearControlOptimizationConfig:
    num_tslots: int = 100
    evo_time: float = 8.8
    max_iter: int = 300
    control_seed: int = 17
    init_control_scale: float = 0.2
    control_bound: float = 2.0
    fidelity_target: float = 0.999
    smoothness_weight: float = 0.0
    amplitude_weight: float = 0.0
    curvature_weight: float = 0.0
    num_restarts: int = 1

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class LinearControlOptimizationResult:
    controls: np.ndarray
    integrated_phases: np.ndarray
    theta: float
    fidelity: float
    objective: float
    iterations: int
    success: bool
    message: str
    evo_time: float
    num_tslots: int
    smoothness_cost: float
    amplitude_cost: float
    curvature_cost: float

    def to_json(self) -> dict[str, float | int | bool | str | list[list[float]]]:
        return {
            "controls": [[float(entry) for entry in row] for row in self.controls],
            "integrated_phases": [[float(entry) for entry in row] for row in self.integrated_phases],
            "theta": float(self.theta),
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "smoothness_cost": float(self.smoothness_cost),
            "amplitude_cost": float(self.amplitude_cost),
            "curvature_cost": float(self.curvature_cost),
        }


@dataclass(frozen=True)
class LinearTimeOptimalScanResult:
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


class LinearControlGRAPEOptimizer:
    """GRAPE optimizer for linearly controlled phase-gate Hamiltonians."""

    def __init__(self, model: LinearPhaseGateModel, config: LinearControlOptimizationConfig):
        self.model = model
        self.config = config
        self.h_d = model.drift_hamiltonian().full()
        self.control_ops = [operator.full() for operator in model.control_hamiltonians()]
        self.initial_state = model.initial_state().full().ravel()
        self.dimension = self.initial_state.shape[0]
        self.num_controls = len(self.control_ops)

    def initial_controls(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.control_seed)
        return rng.normal(
            0.0,
            self.config.init_control_scale,
            size=(self.num_controls, self.config.num_tslots),
        )

    def optimize(
        self,
        initial_controls: np.ndarray | None = None,
        initial_theta: float = 0.0,
    ) -> LinearControlOptimizationResult:
        best_result: LinearControlOptimizationResult | None = None
        base_controls = (
            self.initial_controls()
            if initial_controls is None
            else self._coerce_control_matrix(initial_controls)
        )

        bounds = [(-self.config.control_bound, self.config.control_bound)] * (
            self.num_controls * self.config.num_tslots
        ) + [(0.0, 2.0 * np.pi)]

        for restart in range(self.config.num_restarts):
            controls0 = base_controls if restart == 0 else self._jittered_initial_controls(base_controls, restart)
            variables0 = np.concatenate([controls0.ravel(), np.array([initial_theta])])
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.config.max_iter},
            )

            controls = result.x[:-1].reshape(self.num_controls, self.config.num_tslots)
            theta = float(np.mod(result.x[-1], 2.0 * np.pi))
            final_state = self.final_state(controls)
            fidelity = self.model.phase_gate_fidelity(final_state, theta)
            smoothness_cost = self._smoothness_cost(controls)
            amplitude_cost = self._amplitude_cost(controls)
            curvature_cost = self._curvature_cost(controls)

            candidate = LinearControlOptimizationResult(
                controls=controls,
                integrated_phases=self.integrated_phases(controls),
                theta=theta,
                fidelity=float(fidelity),
                objective=float(
                    1.0
                    - fidelity
                    + self.config.smoothness_weight * smoothness_cost
                    + self.config.amplitude_weight * amplitude_cost
                    + self.config.curvature_weight * curvature_cost
                ),
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
                evo_time=float(self.config.evo_time),
                num_tslots=int(self.config.num_tslots),
                smoothness_cost=float(smoothness_cost),
                amplitude_cost=float(amplitude_cost),
                curvature_cost=float(curvature_cost),
            )
            if best_result is None or self._is_better(candidate, best_result):
                best_result = candidate

        assert best_result is not None
        return best_result

    def scan_durations(
        self,
        durations: list[float],
        initial_controls: np.ndarray | None = None,
    ) -> tuple[LinearTimeOptimalScanResult, list[LinearControlOptimizationResult]]:
        controls = self.initial_controls() if initial_controls is None else self._coerce_control_matrix(initial_controls)
        theta = 0.0
        results: list[LinearControlOptimizationResult] = []
        fidelities: list[float] = []

        for duration in durations:
            optimizer = LinearControlGRAPEOptimizer(
                self.model,
                LinearControlOptimizationConfig(
                    num_tslots=self.config.num_tslots,
                    evo_time=duration,
                    max_iter=self.config.max_iter,
                    control_seed=self.config.control_seed,
                    init_control_scale=self.config.init_control_scale,
                    control_bound=self.config.control_bound,
                    fidelity_target=self.config.fidelity_target,
                    smoothness_weight=self.config.smoothness_weight,
                    amplitude_weight=self.config.amplitude_weight,
                    curvature_weight=self.config.curvature_weight,
                    num_restarts=self.config.num_restarts,
                ),
            )
            result = optimizer.optimize(controls, theta)
            results.append(result)
            fidelities.append(result.fidelity)
            controls = result.controls
            theta = result.theta

        qualified = [res for res in results if res.fidelity >= self.config.fidelity_target]
        best_duration = None if not qualified else min(res.evo_time for res in qualified)

        return (
            LinearTimeOptimalScanResult(
                durations=list(durations),
                fidelities=fidelities,
                best_duration=best_duration,
                best_fidelity=max(fidelities) if fidelities else 0.0,
                target_reached=best_duration is not None,
            ),
            results,
        )

    def save_result(self, result: LinearControlOptimizationResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def save_scan(self, result: LinearTimeOptimalScanResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def integrated_phases(self, controls: np.ndarray) -> np.ndarray:
        control_matrix = self._coerce_control_matrix(controls)
        return np.cumsum(control_matrix, axis=1) * self.config.dt

    def final_state(self, controls: np.ndarray) -> np.ndarray:
        control_matrix = self._coerce_control_matrix(controls)
        state = np.array(self.initial_state, copy=True)
        dt = self.config.dt
        for slot_index in range(self.config.num_tslots):
            u_k = expm(-1j * dt * self._hamiltonian_from_controls(control_matrix[:, slot_index]))
            state = u_k @ state
        return state

    def trajectory(self, controls: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        control_matrix = self._coerce_control_matrix(controls)
        state = np.array(self.initial_state, copy=True)
        states = [state]
        times = np.linspace(0.0, self.config.evo_time, self.config.num_tslots + 1)
        dt = self.config.dt
        for slot_index in range(self.config.num_tslots):
            u_k = expm(-1j * dt * self._hamiltonian_from_controls(control_matrix[:, slot_index]))
            state = u_k @ state
            states.append(state)
        return times, states

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        controls = np.asarray(variables[:-1], dtype=np.float64).reshape(
            self.num_controls,
            self.config.num_tslots,
        )
        theta = float(variables[-1])
        dt = self.config.dt

        slice_unitaries: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_state]
        current_state = np.array(self.initial_state, copy=True)

        for slot_index in range(self.config.num_tslots):
            h_k = self._hamiltonian_from_controls(controls[:, slot_index])
            u_k = expm(-1j * dt * h_k)
            slice_unitaries.append(u_k)
            current_state = u_k @ current_state
            state_prefix.append(current_state)

        final_state = state_prefix[-1]
        alpha = final_state[0]
        beta = final_state[3]
        s = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta

        fidelity = self.model.phase_gate_fidelity(final_state, theta)
        smoothness_cost = self._smoothness_cost(controls)
        amplitude_cost = self._amplitude_cost(controls)
        curvature_cost = self._curvature_cost(controls)
        objective = (
            1.0
            - fidelity
            + self.config.smoothness_weight * smoothness_cost
            + self.config.amplitude_weight * amplitude_cost
            + self.config.curvature_weight * curvature_cost
        )

        suffix_unitaries: list[np.ndarray] = [
            np.eye(self.dimension, dtype=np.complex128) for _ in range(self.config.num_tslots)
        ]
        current_suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(self.config.num_tslots - 1, -1, -1):
            suffix_unitaries[index] = current_suffix
            current_suffix = current_suffix @ slice_unitaries[index]

        control_gradient = np.zeros_like(controls)
        for slot_index in range(self.config.num_tslots):
            h_k = self._hamiltonian_from_controls(controls[:, slot_index])
            for control_index in range(self.num_controls):
                du_k = expm_frechet(
                    -1j * dt * h_k,
                    -1j * dt * self.control_ops[control_index],
                    compute_expm=False,
                )
                d_state = suffix_unitaries[slot_index] @ du_k @ state_prefix[slot_index]
                d_alpha = d_state[0]
                d_beta = d_state[3]
                d_s = 2.0 * np.exp(-1j * theta) * d_alpha - np.exp(-2j * theta) * d_beta
                d_pop = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
                d_fidelity = (2.0 * np.real(np.conj(s) * d_s) + d_pop) / 20.0
                control_gradient[control_index, slot_index] = -d_fidelity

        d_s_theta = -2j * np.exp(-1j * theta) * alpha + 2j * np.exp(-2j * theta) * beta
        d_fidelity_theta = 2.0 * np.real(np.conj(s) * d_s_theta) / 20.0

        if self.config.smoothness_weight > 0.0:
            control_gradient += self.config.smoothness_weight * self._smoothness_gradient(controls)
        if self.config.amplitude_weight > 0.0:
            control_gradient += self.config.amplitude_weight * self._amplitude_gradient(controls)
        if self.config.curvature_weight > 0.0:
            control_gradient += self.config.curvature_weight * self._curvature_gradient(controls)

        gradient = np.concatenate([control_gradient.ravel(), np.array([-d_fidelity_theta])])
        return float(objective), gradient

    def _hamiltonian_from_controls(self, controls: np.ndarray) -> np.ndarray:
        hamiltonian = np.array(self.h_d, copy=True)
        for index, value in enumerate(controls):
            hamiltonian += float(value) * self.control_ops[index]
        return hamiltonian

    def _jittered_initial_controls(self, base_controls: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.control_seed + 1000 * restart)
        jitter = rng.normal(0.0, self.config.init_control_scale * 0.35, size=base_controls.shape)
        return np.clip(base_controls + jitter, -self.config.control_bound, self.config.control_bound)

    def _coerce_control_matrix(self, controls: np.ndarray) -> np.ndarray:
        array = np.asarray(controls, dtype=np.float64)
        if array.shape != (self.num_controls, self.config.num_tslots):
            raise ValueError("Control array shape does not match optimizer configuration")
        return array

    @staticmethod
    def _is_better(left: LinearControlOptimizationResult, right: LinearControlOptimizationResult) -> bool:
        if left.fidelity > right.fidelity + 1e-8:
            return True
        if abs(left.fidelity - right.fidelity) <= 1e-8 and left.objective < right.objective:
            return True
        return False

    @staticmethod
    def _smoothness_cost(controls: np.ndarray) -> float:
        if controls.shape[1] < 2:
            return 0.0
        delta = controls[:, 1:] - controls[:, :-1]
        return float(np.mean(delta**2))

    @staticmethod
    def _smoothness_gradient(controls: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(controls)
        if controls.shape[1] < 2:
            return grad
        scale = 2.0 / (controls.shape[1] - 1)
        delta = controls[:, 1:] - controls[:, :-1]
        grad[:, 0] = -delta[:, 0] * scale
        grad[:, -1] = delta[:, -1] * scale
        if controls.shape[1] > 2:
            grad[:, 1:-1] = (delta[:, :-1] - delta[:, 1:]) * scale
        return grad

    @staticmethod
    def _amplitude_cost(controls: np.ndarray) -> float:
        return float(np.mean(controls**2))

    @staticmethod
    def _amplitude_gradient(controls: np.ndarray) -> np.ndarray:
        return 2.0 * controls / controls.size

    @staticmethod
    def _curvature_cost(controls: np.ndarray) -> float:
        if controls.shape[1] < 3:
            return 0.0
        curvature = controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
        return float(np.mean(curvature**2))

    @staticmethod
    def _curvature_gradient(controls: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(controls)
        if controls.shape[1] < 3:
            return grad
        curvature = controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
        scale = 2.0 / curvature.size
        grad[:, :-2] += curvature * scale
        grad[:, 1:-1] += -2.0 * curvature * scale
        grad[:, 2:] += curvature * scale
        return grad
