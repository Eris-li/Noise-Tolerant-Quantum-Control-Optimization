from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize


@dataclass(frozen=True)
class AmplitudePhaseOptimizationConfig:
    num_tslots: int = 100
    evo_time: float = 8.8
    max_iter: int = 250
    seed: int = 41
    init_phase_spread: float = 0.35
    init_amplitude_scale: float = 0.75
    fidelity_target: float = 0.9999
    phase_smoothness_weight: float = 0.01
    phase_curvature_weight: float = 0.02
    amplitude_smoothness_weight: float = 0.01
    amplitude_curvature_weight: float = 0.01
    num_restarts: int = 3
    show_progress: bool = False

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class AmplitudePhaseOptimizationResult:
    amplitudes: np.ndarray
    phases: np.ndarray
    theta: float
    fidelity: float
    objective: float
    iterations: int
    success: bool
    message: str
    evo_time: float
    num_tslots: int
    phase_smoothness_cost: float
    phase_curvature_cost: float
    amplitude_smoothness_cost: float
    amplitude_curvature_cost: float

    def to_json(self) -> dict[str, float | int | bool | str | list[float]]:
        return {
            "amplitudes": [float(x) for x in self.amplitudes],
            "phases": [float(x) for x in self.phases],
            "theta": float(self.theta),
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "phase_smoothness_cost": float(self.phase_smoothness_cost),
            "phase_curvature_cost": float(self.phase_curvature_cost),
            "amplitude_smoothness_cost": float(self.amplitude_smoothness_cost),
            "amplitude_curvature_cost": float(self.amplitude_curvature_cost),
        }


@dataclass(frozen=True)
class AmplitudePhaseScanResult:
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


class AmplitudePhaseOptimizer:
    """Single-phase, single-amplitude optimizer for the lower leg."""

    def __init__(self, model, config: AmplitudePhaseOptimizationConfig):
        self.model = model
        self.config = config
        self.h_d = model.drift_hamiltonian().full()
        groups = model.phase_control_hamiltonians()
        if len(groups) != 1:
            raise ValueError("AmplitudePhaseOptimizer expects a single optimized control channel")
        self.h_x, self.h_y = [operator.full() for operator in groups[0]]
        self.initial_state = model.initial_state().full().ravel()
        self.phase_gate_indices = model.phase_gate_state_indices()
        self.amplitude_max = float(model.phase_control_amplitudes()[0])

    def initial_guess(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        amplitudes = self.amplitude_max * np.clip(
            self.config.init_amplitude_scale + 0.08 * rng.normal(size=self.config.num_tslots),
            0.0,
            1.0,
        )
        phases = np.mod(
            rng.normal(0.0, self.config.init_phase_spread, size=self.config.num_tslots),
            2.0 * np.pi,
        )
        return amplitudes, phases

    def optimize(
        self,
        initial_amplitudes: np.ndarray | None = None,
        initial_phases: np.ndarray | None = None,
        initial_theta: float = 0.0,
    ) -> AmplitudePhaseOptimizationResult:
        best: AmplitudePhaseOptimizationResult | None = None
        base_amplitudes, base_phases = self.initial_guess()
        if initial_amplitudes is not None:
            base_amplitudes = np.asarray(initial_amplitudes, dtype=np.float64)
        if initial_phases is not None:
            base_phases = np.asarray(initial_phases, dtype=np.float64)

        bounds = (
            [(0.0, self.amplitude_max)] * self.config.num_tslots
            + [(0.0, 2.0 * np.pi)] * self.config.num_tslots
            + [(0.0, 2.0 * np.pi)]
        )

        for restart in range(self.config.num_restarts):
            amplitudes0 = base_amplitudes if restart == 0 else self._jitter_amplitudes(base_amplitudes, restart)
            phases0 = base_phases if restart == 0 else self._jitter_phases(base_phases, restart)
            variables0 = np.concatenate([amplitudes0, phases0, np.array([initial_theta])])
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.config.max_iter},
            )
            amplitudes, phases, theta = self._unpack(result.x)
            final_state = self.final_state(amplitudes, phases)
            fidelity = self.model.phase_gate_fidelity(final_state, theta)
            candidate = AmplitudePhaseOptimizationResult(
                amplitudes=amplitudes,
                phases=phases,
                theta=theta,
                fidelity=float(fidelity),
                objective=float(self.objective_and_gradient(result.x)[0]),
                iterations=int(result.nit),
                success=bool(result.success),
                message=str(result.message),
                evo_time=float(self.config.evo_time),
                num_tslots=int(self.config.num_tslots),
                phase_smoothness_cost=float(self._phase_smoothness_cost(phases)),
                phase_curvature_cost=float(self._phase_curvature_cost(phases)),
                amplitude_smoothness_cost=float(self._amplitude_smoothness_cost(amplitudes)),
                amplitude_curvature_cost=float(self._amplitude_curvature_cost(amplitudes)),
            )
            if best is None or self._is_better(candidate, best):
                best = candidate
        assert best is not None
        return best

    def scan_durations(
        self,
        durations: list[float],
        initial_amplitudes: np.ndarray | None = None,
        initial_phases: np.ndarray | None = None,
    ) -> tuple[AmplitudePhaseScanResult, list[AmplitudePhaseOptimizationResult]]:
        amplitudes, phases = self.initial_guess()
        if initial_amplitudes is not None:
            amplitudes = np.asarray(initial_amplitudes, dtype=np.float64)
        if initial_phases is not None:
            phases = np.asarray(initial_phases, dtype=np.float64)
        theta = 0.0
        results: list[AmplitudePhaseOptimizationResult] = []
        fidelities: list[float] = []
        scan_started_at = time.perf_counter()

        for index, duration in enumerate(durations, start=1):
            started_at = time.perf_counter()
            optimizer = AmplitudePhaseOptimizer(
                self.model,
                AmplitudePhaseOptimizationConfig(
                    num_tslots=self.config.num_tslots,
                    evo_time=duration,
                    max_iter=self.config.max_iter,
                    seed=self.config.seed,
                    init_phase_spread=self.config.init_phase_spread,
                    init_amplitude_scale=self.config.init_amplitude_scale,
                    fidelity_target=self.config.fidelity_target,
                    phase_smoothness_weight=self.config.phase_smoothness_weight,
                    phase_curvature_weight=self.config.phase_curvature_weight,
                    amplitude_smoothness_weight=self.config.amplitude_smoothness_weight,
                    amplitude_curvature_weight=self.config.amplitude_curvature_weight,
                    num_restarts=self.config.num_restarts,
                    show_progress=False,
                ),
            )
            result = optimizer.optimize(amplitudes, phases, theta)
            results.append(result)
            fidelities.append(result.fidelity)
            amplitudes, phases, theta = result.amplitudes, result.phases, result.theta
            if self.config.show_progress:
                step_elapsed = time.perf_counter() - started_at
                total_elapsed = time.perf_counter() - scan_started_at
                avg_per_step = total_elapsed / index
                remaining = avg_per_step * (len(durations) - index)
                print(
                    f"[scan] {index}/{len(durations)} ({100.0*index/len(durations):5.1f}%) "
                    f"T={duration:.3f} F={result.fidelity:.9f} "
                    f"step={step_elapsed:7.1f}s elapsed={total_elapsed:7.1f}s "
                    f"remaining_T={len(durations)-index:2d} eta={remaining:7.1f}s",
                    flush=True,
                )

        qualified = [res for res in results if res.fidelity >= self.config.fidelity_target]
        best_duration = None if not qualified else min(res.evo_time for res in qualified)
        return (
            AmplitudePhaseScanResult(
                durations=list(durations),
                fidelities=fidelities,
                best_duration=best_duration,
                best_fidelity=max(fidelities) if fidelities else 0.0,
                target_reached=best_duration is not None,
            ),
            results,
        )

    def save_result(self, result: AmplitudePhaseOptimizationResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def save_scan(self, result: AmplitudePhaseScanResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def final_state(self, amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
        state = np.array(self.initial_state, copy=True)
        dt = self.config.dt
        for amplitude, phase in zip(amplitudes, phases):
            u_k = expm(-1j * dt * self._hamiltonian(float(amplitude), float(phase)))
            state = u_k @ state
        return state

    def trajectory(self, amplitudes: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        state = np.array(self.initial_state, copy=True)
        states = [state]
        times = np.linspace(0.0, self.config.evo_time, self.config.num_tslots + 1)
        dt = self.config.dt
        for amplitude, phase in zip(amplitudes, phases):
            u_k = expm(-1j * dt * self._hamiltonian(float(amplitude), float(phase)))
            state = u_k @ state
            states.append(state)
        return times, states

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        amplitudes, phases, theta = self._unpack(variables)
        dt = self.config.dt
        slice_unitaries: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_state]
        current_state = np.array(self.initial_state, copy=True)

        for amplitude, phase in zip(amplitudes, phases):
            h_k = self._hamiltonian(float(amplitude), float(phase))
            u_k = expm(-1j * dt * h_k)
            slice_unitaries.append(u_k)
            current_state = u_k @ current_state
            state_prefix.append(current_state)

        final_state = state_prefix[-1]
        alpha = final_state[self.phase_gate_indices[0]]
        beta = final_state[self.phase_gate_indices[1]]
        s = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
        fidelity = self.model.phase_gate_fidelity(final_state, theta)

        phase_smoothness = self._phase_smoothness_cost(phases)
        phase_curvature = self._phase_curvature_cost(phases)
        amplitude_smoothness = self._amplitude_smoothness_cost(amplitudes)
        amplitude_curvature = self._amplitude_curvature_cost(amplitudes)
        objective = (
            1.0 - fidelity
            + self.config.phase_smoothness_weight * phase_smoothness
            + self.config.phase_curvature_weight * phase_curvature
            + self.config.amplitude_smoothness_weight * amplitude_smoothness
            + self.config.amplitude_curvature_weight * amplitude_curvature
        )

        suffix_unitaries: list[np.ndarray] = [np.eye(self.initial_state.shape[0], dtype=np.complex128) for _ in amplitudes]
        current_suffix = np.eye(self.initial_state.shape[0], dtype=np.complex128)
        for index in range(len(amplitudes) - 1, -1, -1):
            suffix_unitaries[index] = current_suffix
            current_suffix = current_suffix @ slice_unitaries[index]

        amplitude_grad = np.zeros_like(amplitudes)
        phase_grad = np.zeros_like(phases)
        for index, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
            h_k = self._hamiltonian(float(amplitude), float(phase))
            d_h_amp = np.cos(phase) * self.h_x + np.sin(phase) * self.h_y
            d_h_phi = amplitude * (-np.sin(phase) * self.h_x + np.cos(phase) * self.h_y)
            for target, d_h in ((amplitude_grad, d_h_amp), (phase_grad, d_h_phi)):
                du_k = expm_frechet(-1j * dt * h_k, -1j * dt * d_h, compute_expm=False)
                d_state = suffix_unitaries[index] @ du_k @ state_prefix[index]
                d_alpha = d_state[self.phase_gate_indices[0]]
                d_beta = d_state[self.phase_gate_indices[1]]
                d_s = 2.0 * np.exp(-1j * theta) * d_alpha - np.exp(-2j * theta) * d_beta
                d_pop = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
                d_fidelity = (2.0 * np.real(np.conj(s) * d_s) + d_pop) / 20.0
                target[index] = -d_fidelity

        d_s_theta = -2j * np.exp(-1j * theta) * alpha + 2j * np.exp(-2j * theta) * beta
        d_fidelity_theta = 2.0 * np.real(np.conj(s) * d_s_theta) / 20.0

        if self.config.phase_smoothness_weight > 0.0:
            phase_grad += self.config.phase_smoothness_weight * self._phase_smoothness_gradient(phases)
        if self.config.phase_curvature_weight > 0.0:
            phase_grad += self.config.phase_curvature_weight * self._phase_curvature_gradient(phases)
        if self.config.amplitude_smoothness_weight > 0.0:
            amplitude_grad += self.config.amplitude_smoothness_weight * self._amplitude_smoothness_gradient(amplitudes)
        if self.config.amplitude_curvature_weight > 0.0:
            amplitude_grad += self.config.amplitude_curvature_weight * self._amplitude_curvature_gradient(amplitudes)

        gradient = np.concatenate([amplitude_grad, phase_grad, np.array([-d_fidelity_theta])])
        return float(objective), gradient

    def _hamiltonian(self, amplitude: float, phase: float) -> np.ndarray:
        return self.h_d + amplitude * (np.cos(phase) * self.h_x + np.sin(phase) * self.h_y)

    def _unpack(self, variables: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        amplitudes = np.asarray(variables[: self.config.num_tslots], dtype=np.float64)
        phases = np.mod(
            np.asarray(variables[self.config.num_tslots : 2 * self.config.num_tslots], dtype=np.float64),
            2.0 * np.pi,
        )
        theta = float(np.mod(variables[-1], 2.0 * np.pi))
        return amplitudes, phases, theta

    def _jitter_amplitudes(self, base: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed + 1000 * restart)
        jitter = 0.08 * self.amplitude_max * rng.normal(size=base.shape[0])
        return np.clip(base + jitter, 0.0, self.amplitude_max)

    def _jitter_phases(self, base: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed + 1000 * restart + 1)
        jitter = rng.normal(0.0, self.config.init_phase_spread * 0.35, size=base.shape[0])
        return np.mod(base + jitter, 2.0 * np.pi)

    @staticmethod
    def _phase_smoothness_cost(phases: np.ndarray) -> float:
        if len(phases) < 2:
            return 0.0
        delta = phases[1:] - phases[:-1]
        return float(np.mean(1.0 - np.cos(delta)))

    @staticmethod
    def _phase_smoothness_gradient(phases: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _phase_curvature_cost(phases: np.ndarray) -> float:
        if len(phases) < 3:
            return 0.0
        delta = np.angle(np.exp(1j * (phases[1:] - phases[:-1])))
        curvature = delta[1:] - delta[:-1]
        return float(np.mean(curvature**2))

    @staticmethod
    def _phase_curvature_gradient(phases: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(phases)
        if len(phases) < 3:
            return grad
        unwrapped = np.unwrap(phases)
        curvature = unwrapped[2:] - 2.0 * unwrapped[1:-1] + unwrapped[:-2]
        scale = 2.0 / curvature.size
        grad[:-2] += curvature * scale
        grad[1:-1] += -2.0 * curvature * scale
        grad[2:] += curvature * scale
        return grad

    @staticmethod
    def _amplitude_smoothness_cost(amplitudes: np.ndarray) -> float:
        if len(amplitudes) < 2:
            return 0.0
        delta = amplitudes[1:] - amplitudes[:-1]
        return float(np.mean(delta**2))

    @staticmethod
    def _amplitude_smoothness_gradient(amplitudes: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(amplitudes)
        if len(amplitudes) < 2:
            return grad
        scale = 2.0 / (len(amplitudes) - 1)
        delta = amplitudes[1:] - amplitudes[:-1]
        grad[0] = -delta[0] * scale
        grad[-1] = delta[-1] * scale
        if len(amplitudes) > 2:
            grad[1:-1] = (delta[:-1] - delta[1:]) * scale
        return grad

    @staticmethod
    def _amplitude_curvature_cost(amplitudes: np.ndarray) -> float:
        if len(amplitudes) < 3:
            return 0.0
        curvature = amplitudes[2:] - 2.0 * amplitudes[1:-1] + amplitudes[:-2]
        return float(np.mean(curvature**2))

    @staticmethod
    def _amplitude_curvature_gradient(amplitudes: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(amplitudes)
        if len(amplitudes) < 3:
            return grad
        curvature = amplitudes[2:] - 2.0 * amplitudes[1:-1] + amplitudes[:-2]
        scale = 2.0 / curvature.size
        grad[:-2] += curvature * scale
        grad[1:-1] += -2.0 * curvature * scale
        grad[2:] += curvature * scale
        return grad

    @staticmethod
    def _is_better(left: AmplitudePhaseOptimizationResult, right: AmplitudePhaseOptimizationResult) -> bool:
        if left.fidelity > right.fidelity + 1e-8:
            return True
        if abs(left.fidelity - right.fidelity) <= 1e-8 and left.objective < right.objective:
            return True
        return False
