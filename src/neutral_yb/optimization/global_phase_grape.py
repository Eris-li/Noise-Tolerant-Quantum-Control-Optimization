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
        serialized_phases = (
            [[float(entry) for entry in row] for row in self.phases]
            if np.asarray(self.phases).ndim == 2
            else [float(x) for x in self.phases]
        )
        return {
            "phases": serialized_phases,
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
    """Paper-style optimizer for phase-gate control problems.

    The optimizer supports one or more phase-modulated control channels.
    A single channel corresponds to the original global-phase idealization:
    H(t) = H_d + Omega * [cos(phi) H_x + sin(phi) H_y].

    Multi-channel models can expose several phase-modulated beams, such as a
    two-photon ladder with independent lower- and upper-leg phases.
    """

    def __init__(self, model: PhaseGateModel, config: GlobalPhaseOptimizationConfig):
        self.model = model
        self.config = config
        self.h_d = model.drift_hamiltonian().full()
        self.initial_state = model.initial_state().full().ravel()
        self.dimension = self.initial_state.shape[0]
        self.control_groups = self._extract_control_groups()
        self.num_controls = len(self.control_groups)
        self.control_amplitudes = self._extract_control_amplitudes()

    def initial_phases(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed)
        phases = rng.normal(
            0.0,
            self.config.init_phase_spread,
            size=(self.num_controls, self.config.num_tslots),
        )
        return np.mod(phases, 2.0 * np.pi)

    def optimize(
        self,
        initial_phases: np.ndarray | None = None,
        initial_theta: float = 0.0,
    ) -> GlobalPhaseOptimizationResult:
        best_result: GlobalPhaseOptimizationResult | None = None
        base_phases = (
            self.initial_phases()
            if initial_phases is None
            else self._coerce_phase_matrix(initial_phases)
        )

        for restart in range(self.config.num_restarts):
            phases0 = base_phases if restart == 0 else self._jittered_initial_phases(base_phases, restart)
            variables0 = np.concatenate(
                [np.asarray(phases0, dtype=np.float64).ravel(), np.array([initial_theta])]
            )
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, 2.0 * np.pi)] * len(variables0),
                options={"maxiter": self.config.max_iter},
            )

            phases = np.mod(
                result.x[:-1].reshape(self.num_controls, self.config.num_tslots),
                2.0 * np.pi,
            )
            theta = float(np.mod(result.x[-1], 2.0 * np.pi))
            final_state = self.final_state(phases)
            fidelity = self.model.phase_gate_fidelity(final_state, theta)
            smoothness_cost = self._smoothness_cost(phases)

            candidate = GlobalPhaseOptimizationResult(
                phases=self._phase_output(phases),
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
        phases = self.initial_phases() if initial_phases is None else self._coerce_phase_matrix(initial_phases)
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
        phase_matrix = self._coerce_phase_matrix(phases)
        state = np.array(self.initial_state, copy=True)
        dt = self.config.dt
        for slot_index in range(self.config.num_tslots):
            u_k = expm(-1j * dt * self._hamiltonian_from_phases(phase_matrix[:, slot_index]))
            state = u_k @ state
        return state

    def trajectory(self, phases: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        phase_matrix = self._coerce_phase_matrix(phases)
        state = np.array(self.initial_state, copy=True)
        states = [state]
        times = np.linspace(0.0, self.config.evo_time, self.config.num_tslots + 1)
        dt = self.config.dt
        for slot_index in range(self.config.num_tslots):
            u_k = expm(-1j * dt * self._hamiltonian_from_phases(phase_matrix[:, slot_index]))
            state = u_k @ state
            states.append(state)
        return times, states

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        phases = np.asarray(variables[:-1], dtype=np.float64).reshape(
            self.num_controls,
            self.config.num_tslots,
        )
        theta = float(variables[-1])
        dt = self.config.dt

        slice_unitaries: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_state]
        current_state = np.array(self.initial_state, copy=True)

        for slot_index in range(self.config.num_tslots):
            h_k = self._hamiltonian_from_phases(phases[:, slot_index])
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

        suffix_unitaries: list[np.ndarray] = [
            np.eye(self.dimension, dtype=np.complex128) for _ in range(self.config.num_tslots)
        ]
        current_suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(self.config.num_tslots - 1, -1, -1):
            suffix_unitaries[index] = current_suffix
            current_suffix = current_suffix @ slice_unitaries[index]

        phase_gradient = np.zeros_like(phases)

        for slot_index in range(self.config.num_tslots):
            h_k = self._hamiltonian_from_phases(phases[:, slot_index])
            for control_index in range(self.num_controls):
                dh_k = self._phase_derivative_hamiltonian(
                    control_index,
                    float(phases[control_index, slot_index]),
                )
                du_k = expm_frechet(-1j * dt * h_k, -1j * dt * dh_k, compute_expm=False)
                d_state = suffix_unitaries[slot_index] @ du_k @ state_prefix[slot_index]

                d_alpha = d_state[0]
                d_beta = d_state[2]
                d_s = 2.0 * np.exp(-1j * theta) * d_alpha - np.exp(-2j * theta) * d_beta
                d_pop = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
                d_fidelity = (2.0 * np.real(np.conj(s) * d_s) + d_pop) / 20.0
                phase_gradient[control_index, slot_index] = -d_fidelity

        d_s_theta = -2j * np.exp(-1j * theta) * alpha + 2j * np.exp(-2j * theta) * beta
        d_fidelity_theta = 2.0 * np.real(np.conj(s) * d_s_theta) / 20.0

        if self.config.smoothness_weight > 0.0:
            phase_gradient += self.config.smoothness_weight * self._smoothness_gradient(phases)

        gradient = np.concatenate([phase_gradient.ravel(), np.array([-d_fidelity_theta])])

        return float(objective), gradient

    def _hamiltonian_from_phases(self, phases: np.ndarray) -> np.ndarray:
        hamiltonian = np.array(self.h_d, copy=True)
        for control_index, phase in enumerate(phases):
            h_x, h_y = self.control_groups[control_index]
            amplitude = self.control_amplitudes[control_index]
            hamiltonian += amplitude * (
                np.cos(float(phase)) * h_x + np.sin(float(phase)) * h_y
            )
        return hamiltonian

    def _phase_derivative_hamiltonian(self, control_index: int, phase: float) -> np.ndarray:
        h_x, h_y = self.control_groups[control_index]
        amplitude = self.control_amplitudes[control_index]
        return amplitude * (-np.sin(phase) * h_x + np.cos(phase) * h_y)

    def _jittered_initial_phases(self, base_phases: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed + 1000 * restart)
        jitter = rng.normal(0.0, self.config.init_phase_spread * 0.35, size=base_phases.shape)
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
        phase_matrix = phases if np.asarray(phases).ndim == 2 else np.asarray(phases)[None, :]
        if phase_matrix.shape[1] < 2:
            return 0.0
        delta = phase_matrix[:, 1:] - phase_matrix[:, :-1]
        return float(np.mean(1.0 - np.cos(delta)))

    @staticmethod
    def _smoothness_gradient(phases: np.ndarray) -> np.ndarray:
        phase_matrix = phases if np.asarray(phases).ndim == 2 else np.asarray(phases)[None, :]
        grad = np.zeros_like(phase_matrix)
        if phase_matrix.shape[1] < 2:
            return grad
        scale = 1.0 / (phase_matrix.shape[1] - 1)
        delta = phase_matrix[:, 1:] - phase_matrix[:, :-1]
        grad[:, 0] = -np.sin(delta[:, 0]) * scale
        grad[:, -1] = np.sin(delta[:, -1]) * scale
        if phase_matrix.shape[1] > 2:
            grad[:, 1:-1] = (np.sin(delta[:, :-1]) - np.sin(delta[:, 1:])) * scale
        return grad

    def _extract_control_groups(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if hasattr(self.model, "phase_control_hamiltonians"):
            groups = getattr(self.model, "phase_control_hamiltonians")()
        else:
            groups = (self.model.control_hamiltonians(),)
        return [
            (np.asarray(h_x.full(), dtype=np.complex128), np.asarray(h_y.full(), dtype=np.complex128))
            for h_x, h_y in groups
        ]

    def _extract_control_amplitudes(self) -> list[float]:
        if hasattr(self.model, "phase_control_amplitudes"):
            amplitudes = tuple(float(value) for value in getattr(self.model, "phase_control_amplitudes")())
            if len(amplitudes) != self.num_controls:
                raise ValueError("phase_control_amplitudes must match the number of control groups")
            return list(amplitudes)
        return [float(self.model.species.omega_max)] * self.num_controls

    def _coerce_phase_matrix(self, phases: np.ndarray) -> np.ndarray:
        array = np.asarray(phases, dtype=np.float64)
        if array.ndim == 1:
            if self.num_controls != 1 or array.shape[0] != self.config.num_tslots:
                raise ValueError("Expected a 2D phase array for multi-control optimization")
            return array.reshape(1, self.config.num_tslots)
        if array.shape != (self.num_controls, self.config.num_tslots):
            raise ValueError("Phase array shape does not match optimizer configuration")
        return array

    def _phase_output(self, phases: np.ndarray) -> np.ndarray:
        return phases[0].copy() if self.num_controls == 1 else phases.copy()
