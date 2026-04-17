from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import qutip
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize


@dataclass(frozen=True)
class OpenSystemGRAPEConfig:
    num_tslots: int = 24
    evo_time: float = 8.8
    max_iter: int = 120
    max_wall_time: float = 600.0
    fid_err_targ: float = 5e-3
    min_grad: float = 1e-8
    num_restarts: int = 3
    seed: int = 17
    init_pulse_type: str = "SINE"
    init_control_scale: float = 0.15
    control_smoothness_weight: float = 1e-3
    control_curvature_weight: float = 2e-3
    target_theta: float = 0.0
    fidelity_target: float = 0.995
    show_progress: bool = False

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class OpenSystemGRAPEResult:
    ctrl_x: np.ndarray
    ctrl_y: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    target_theta: float
    optimized_theta: float
    fid_err: float
    probe_fidelity: float
    objective: float
    num_iter: int
    num_fid_func_calls: int
    wall_time: float
    termination_reason: str
    evo_time: float
    num_tslots: int
    control_smoothness_cost: float
    control_curvature_cost: float
    success: bool

    def to_json(self) -> dict[str, float | int | str | bool | list[float]]:
        return {
            "ctrl_x": [float(x) for x in self.ctrl_x],
            "ctrl_y": [float(x) for x in self.ctrl_y],
            "amplitudes": [float(x) for x in self.amplitudes],
            "phases": [float(x) for x in self.phases],
            "target_theta": float(self.target_theta),
            "optimized_theta": float(self.optimized_theta),
            "fid_err": float(self.fid_err),
            "probe_fidelity": float(self.probe_fidelity),
            "objective": float(self.objective),
            "num_iter": int(self.num_iter),
            "num_fid_func_calls": int(self.num_fid_func_calls),
            "wall_time": float(self.wall_time),
            "termination_reason": self.termination_reason,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "control_smoothness_cost": float(self.control_smoothness_cost),
            "control_curvature_cost": float(self.control_curvature_cost),
            "success": bool(self.success),
        }


@dataclass(frozen=True)
class OpenSystemScanResult:
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


class OpenSystemGRAPEOptimizer:
    r"""Open-system GRAPE aligned with the CZ fidelity from arXiv:2202.00903.

    The optimization objective uses the two-qubit phase-gate fidelity for the
    reduced |01>, |11> active manifold:

        F = (|1 + 2 a01 + a11|^2 + 1 + 2|a01|^2 + |a11|^2) / 20

    with a01 = exp(-i theta) <01|psi(T)> and
         a11 = -exp(-2 i theta) <11|psi(T)>.

    In the open-system setting we propagate the left coherence amplitudes
    against an ideal spectator branch using an effective non-Hermitian
    generator G = -i H - 1/2 sum_k C_k^\dagger C_k. This preserves the phase
    information required by the fidelity while remaining compatible with the
    Lindblad model used elsewhere for density-matrix analysis.
    """

    def __init__(self, model, config: OpenSystemGRAPEConfig):
        self.model = model
        self.config = config
        self.dimension = int(model.dimension())
        self.vector_dimension = self.dimension * self.dimension
        self.active_indices = tuple(int(index) for index in model.active_gate_indices())
        self.amp_bound = float(self.model.control_amplitude_bound())
        self.h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
        self.h_x, self.h_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in model.lower_leg_control_hamiltonians()
        ]
        decay_matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for operator in model.collapse_operators():
            c_matrix = np.asarray(operator.full(), dtype=np.complex128)
            decay_matrix += c_matrix.conj().T @ c_matrix
        self.g_d = -1j * self.h_d - 0.5 * decay_matrix
        self.g_x = -1j * self.h_x
        self.g_y = -1j * self.h_y
        self.l_d = np.asarray(model.drift_liouvillian().full(), dtype=np.complex128)
        self.l_x, self.l_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in model.control_liouvillians()
        ]
        self.initial_phase_state = np.asarray(
            model.special_phase_gate_state().full(),
            dtype=np.complex128,
        ).ravel()

    def initial_guess(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        slots = self.config.num_tslots
        scale = self.config.init_control_scale * self.amp_bound
        pulse_type = self.config.init_pulse_type.upper()
        grid = np.linspace(0.0, np.pi, slots, endpoint=False, dtype=np.float64)

        if pulse_type == "ZERO":
            ctrl_x = np.zeros(slots, dtype=np.float64)
            ctrl_y = np.zeros(slots, dtype=np.float64)
        elif pulse_type == "RANDOM":
            ctrl_x = scale * rng.normal(size=slots)
            ctrl_y = scale * rng.normal(size=slots)
        else:
            phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
            phase_y = float(rng.uniform(0.0, 2.0 * np.pi))
            envelope = np.sin(grid + 0.5 * np.pi / slots)
            ctrl_x = scale * envelope * np.cos(grid + phase_x)
            ctrl_y = scale * envelope * np.sin(grid + phase_y)

        return (
            np.clip(ctrl_x, -self.amp_bound, self.amp_bound),
            np.clip(ctrl_y, -self.amp_bound, self.amp_bound),
        )

    def optimize(
        self,
        initial_ctrl_x: np.ndarray | None = None,
        initial_ctrl_y: np.ndarray | None = None,
        initial_theta: float | None = None,
    ) -> OpenSystemGRAPEResult:
        best: OpenSystemGRAPEResult | None = None
        base_ctrl_x, base_ctrl_y = self.initial_guess()
        if initial_ctrl_x is not None:
            base_ctrl_x = np.asarray(initial_ctrl_x, dtype=np.float64)
        if initial_ctrl_y is not None:
            base_ctrl_y = np.asarray(initial_ctrl_y, dtype=np.float64)
        theta0 = float(self.config.target_theta if initial_theta is None else initial_theta)

        bounds = (
            [(-self.amp_bound, self.amp_bound)] * self.config.num_tslots
            + [(-self.amp_bound, self.amp_bound)] * self.config.num_tslots
            + [(0.0, 2.0 * np.pi)]
        )

        for restart in range(self.config.num_restarts):
            ctrl_x0 = base_ctrl_x if restart == 0 else self._jitter_controls(base_ctrl_x, restart, axis=0)
            ctrl_y0 = base_ctrl_y if restart == 0 else self._jitter_controls(base_ctrl_y, restart, axis=1)
            variables0 = np.concatenate([ctrl_x0, ctrl_y0, np.array([theta0], dtype=np.float64)])

            if self.config.show_progress:
                print(
                    f"[open-grape] restart {restart + 1}/{self.config.num_restarts} "
                    f"T={self.config.evo_time:.3f} slots={self.config.num_tslots}",
                    flush=True,
                )

            started_at = time.perf_counter()
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.config.max_iter},
            )
            wall_time = time.perf_counter() - started_at
            candidate = self._result_from_variables(
                result.x,
                num_iter=int(result.nit),
                num_fid_func_calls=int(result.nfev),
                wall_time=wall_time,
                termination_reason=str(result.message),
                success=bool(result.success),
            )
            if self.config.show_progress:
                print(
                    f"[open-grape] restart {restart + 1}/{self.config.num_restarts} "
                    f"F={candidate.probe_fidelity:.6f} obj={candidate.objective:.6e} "
                    f"time={candidate.wall_time:7.1f}s",
                    flush=True,
                )
            if best is None or self._is_better(candidate, best):
                best = candidate

        assert best is not None
        return best

    def scan_durations(
        self,
        durations: list[float],
        initial_ctrl_x: np.ndarray | None = None,
        initial_ctrl_y: np.ndarray | None = None,
        initial_theta: float | None = None,
    ) -> tuple[OpenSystemScanResult, list[OpenSystemGRAPEResult]]:
        ctrl_x, ctrl_y = self.initial_guess()
        if initial_ctrl_x is not None:
            ctrl_x = np.asarray(initial_ctrl_x, dtype=np.float64)
        if initial_ctrl_y is not None:
            ctrl_y = np.asarray(initial_ctrl_y, dtype=np.float64)
        theta = float(self.config.target_theta if initial_theta is None else initial_theta)

        results: list[OpenSystemGRAPEResult] = []
        fidelities: list[float] = []
        scan_started_at = time.perf_counter()

        for index, duration in enumerate(durations, start=1):
            started_at = time.perf_counter()
            if self.config.show_progress:
                print(
                    f"[scan] starting {index}/{len(durations)} "
                    f"T={duration:.3f} slots={self.config.num_tslots}",
                    flush=True,
                )
            optimizer = OpenSystemGRAPEOptimizer(
                self.model,
                OpenSystemGRAPEConfig(
                    num_tslots=self.config.num_tslots,
                    evo_time=duration,
                    max_iter=self.config.max_iter,
                    max_wall_time=self.config.max_wall_time,
                    fid_err_targ=self.config.fid_err_targ,
                    min_grad=self.config.min_grad,
                    num_restarts=self.config.num_restarts,
                    seed=self.config.seed,
                    init_pulse_type=self.config.init_pulse_type,
                    init_control_scale=self.config.init_control_scale,
                    control_smoothness_weight=self.config.control_smoothness_weight,
                    control_curvature_weight=self.config.control_curvature_weight,
                    target_theta=self.config.target_theta,
                    fidelity_target=self.config.fidelity_target,
                    show_progress=self.config.show_progress,
                ),
            )
            result = optimizer.optimize(
                initial_ctrl_x=ctrl_x,
                initial_ctrl_y=ctrl_y,
                initial_theta=theta,
            )
            results.append(result)
            fidelities.append(result.probe_fidelity)
            ctrl_x, ctrl_y, theta = result.ctrl_x, result.ctrl_y, result.optimized_theta
            if self.config.show_progress:
                step_elapsed = time.perf_counter() - started_at
                total_elapsed = time.perf_counter() - scan_started_at
                avg_per_step = total_elapsed / index
                remaining = avg_per_step * (len(durations) - index)
                print(
                    f"[scan] {index}/{len(durations)} ({100.0*index/len(durations):5.1f}%) "
                    f"T={duration:.3f} F={result.probe_fidelity:.6f} "
                    f"step={step_elapsed:7.1f}s elapsed={total_elapsed:7.1f}s "
                    f"remaining_T={len(durations)-index:2d} eta={remaining:7.1f}s",
                    flush=True,
                )

        qualified = [res for res in results if res.probe_fidelity >= self.config.fidelity_target]
        best_duration = None if not qualified else min(res.evo_time for res in qualified)
        return (
            OpenSystemScanResult(
                durations=list(durations),
                fidelities=fidelities,
                best_duration=best_duration,
                best_fidelity=max(fidelities) if fidelities else 0.0,
                target_reached=best_duration is not None,
            ),
            results,
        )

    def evolve_probe_states(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> list[qutip.Qobj]:
        final_vectors = self._final_probe_vectors(ctrl_x, ctrl_y)
        return [qutip.Qobj(self._vec_to_matrix(vector)) for vector in final_vectors.T]

    def evolve_density_matrix(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        rho0: qutip.Qobj,
    ) -> qutip.Qobj:
        vector = np.asarray(rho0.full(), dtype=np.complex128).reshape(-1, order="F")
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            vector = expm(self.config.dt * self._liouvillian(float(x_value), float(y_value))) @ vector
        return qutip.Qobj(self._vec_to_matrix(vector))

    def trajectory(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        rho0: qutip.Qobj,
    ) -> tuple[np.ndarray, list[qutip.Qobj]]:
        vector = np.asarray(rho0.full(), dtype=np.complex128).reshape(-1, order="F")
        states = [qutip.Qobj(self._vec_to_matrix(vector))]
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            vector = expm(self.config.dt * self._liouvillian(float(x_value), float(y_value))) @ vector
            states.append(qutip.Qobj(self._vec_to_matrix(vector)))
        times = np.linspace(0.0, self.config.evo_time, self.config.num_tslots + 1)
        return times, states

    def benchmark_probe_evolution(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, repeats: int = 3) -> float:
        self.final_phase_state(ctrl_x, ctrl_y)
        started_at = time.perf_counter()
        for _ in range(repeats):
            self.final_phase_state(ctrl_x, ctrl_y)
        return (time.perf_counter() - started_at) / repeats

    def save_result(self, result: OpenSystemGRAPEResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def save_scan(self, result: OpenSystemScanResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        ctrl_x, ctrl_y, theta = self._unpack(variables)
        dt = self.config.dt
        slice_propags: list[np.ndarray] = []
        state_prefix: list[np.ndarray] = [self.initial_phase_state]
        current_state = np.array(self.initial_phase_state, copy=True)

        for x_value, y_value in zip(ctrl_x, ctrl_y):
            g_k = self._effective_generator(float(x_value), float(y_value))
            u_k = expm(dt * g_k)
            slice_propags.append(u_k)
            current_state = u_k @ current_state
            state_prefix.append(current_state)

        final_state = state_prefix[-1]
        fidelity, fidelity_theta_grad = self._phase_gate_fidelity_and_theta_gradient(final_state, theta)
        smoothness_cost = self._control_smoothness_cost(ctrl_x, ctrl_y)
        curvature_cost = self._control_curvature_cost(ctrl_x, ctrl_y)
        objective = (
            1.0 - fidelity
            + self.config.control_smoothness_weight * smoothness_cost
            + self.config.control_curvature_weight * curvature_cost
        )

        suffix_propags: list[np.ndarray] = [np.eye(self.dimension, dtype=np.complex128) for _ in range(self.config.num_tslots)]
        current_suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(self.config.num_tslots - 1, -1, -1):
            suffix_propags[index] = current_suffix
            current_suffix = current_suffix @ slice_propags[index]

        ctrl_x_grad = np.zeros_like(ctrl_x)
        ctrl_y_grad = np.zeros_like(ctrl_y)
        for index, (x_value, y_value) in enumerate(zip(ctrl_x, ctrl_y)):
            g_k = self._effective_generator(float(x_value), float(y_value))
            du_x = expm_frechet(dt * g_k, dt * self.g_x, compute_expm=False)
            du_y = expm_frechet(dt * g_k, dt * self.g_y, compute_expm=False)
            d_state_x = suffix_propags[index] @ du_x @ state_prefix[index]
            d_state_y = suffix_propags[index] @ du_y @ state_prefix[index]
            ctrl_x_grad[index] = -self._phase_gate_fidelity_state_gradient(final_state, d_state_x, theta)
            ctrl_y_grad[index] = -self._phase_gate_fidelity_state_gradient(final_state, d_state_y, theta)

        if self.config.control_smoothness_weight > 0.0:
            ctrl_x_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_y)
        if self.config.control_curvature_weight > 0.0:
            ctrl_x_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_y)

        gradient = np.concatenate([ctrl_x_grad, ctrl_y_grad, np.array([-fidelity_theta_grad])])
        return float(objective), gradient

    def _result_from_variables(
        self,
        variables: np.ndarray,
        num_iter: int,
        num_fid_func_calls: int,
        wall_time: float,
        termination_reason: str,
        success: bool,
    ) -> OpenSystemGRAPEResult:
        ctrl_x, ctrl_y, theta = self._unpack(variables)
        amplitudes, phases = self.model.control_cartesian_to_polar(ctrl_x, ctrl_y)
        final_state = self.final_phase_state(ctrl_x, ctrl_y)
        probe_fidelity, _ = self._phase_gate_fidelity_and_theta_gradient(final_state, theta)
        objective = (
            1.0
            - probe_fidelity
            + self.config.control_smoothness_weight * self._control_smoothness_cost(ctrl_x, ctrl_y)
            + self.config.control_curvature_weight * self._control_curvature_cost(ctrl_x, ctrl_y)
        )
        return OpenSystemGRAPEResult(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            amplitudes=amplitudes,
            phases=phases,
            target_theta=float(self.config.target_theta),
            optimized_theta=float(theta),
            fid_err=float(1.0 - probe_fidelity),
            probe_fidelity=float(probe_fidelity),
            objective=float(objective),
            num_iter=int(num_iter),
            num_fid_func_calls=int(num_fid_func_calls),
            wall_time=float(wall_time),
            termination_reason=termination_reason,
            evo_time=float(self.config.evo_time),
            num_tslots=int(self.config.num_tslots),
            control_smoothness_cost=float(self._control_smoothness_cost(ctrl_x, ctrl_y)),
            control_curvature_cost=float(self._control_curvature_cost(ctrl_x, ctrl_y)),
            success=bool(success),
        )

    def final_phase_state(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> np.ndarray:
        current_state = np.array(self.initial_phase_state, copy=True)
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            current_state = expm(self.config.dt * self._effective_generator(float(x_value), float(y_value))) @ current_state
        return current_state

    def _phase_gate_fidelity_and_theta_gradient(self, final_state: np.ndarray, theta: float) -> tuple[float, float]:
        alpha = complex(final_state[self.active_indices[0]])
        beta = complex(final_state[self.active_indices[1]])
        phase_01 = np.exp(-1j * theta)
        phase_11 = np.exp(-2j * theta)
        phased_sum = 1.0 + 2.0 * phase_01 * alpha - phase_11 * beta
        population_sum = 1.0 + 2.0 * abs(alpha) ** 2 + abs(beta) ** 2
        fidelity = (abs(phased_sum) ** 2 + population_sum) / 20.0
        phased_sum_derivative = -2j * phase_01 * alpha + 2j * phase_11 * beta
        theta_gradient = np.real(np.conj(phased_sum) * phased_sum_derivative) / 10.0
        return float(fidelity), float(theta_gradient)

    def _phase_gate_fidelity_state_gradient(
        self,
        final_state: np.ndarray,
        d_state: np.ndarray,
        theta: float,
    ) -> float:
        alpha = complex(final_state[self.active_indices[0]])
        beta = complex(final_state[self.active_indices[1]])
        d_alpha = complex(d_state[self.active_indices[0]])
        d_beta = complex(d_state[self.active_indices[1]])
        phase_01 = np.exp(-1j * theta)
        phase_11 = np.exp(-2j * theta)
        phased_sum = 1.0 + 2.0 * phase_01 * alpha - phase_11 * beta
        delta_phased_sum = 2.0 * phase_01 * d_alpha - phase_11 * d_beta
        delta_population = 4.0 * np.real(np.conj(alpha) * d_alpha) + 2.0 * np.real(np.conj(beta) * d_beta)
        delta_fidelity = (2.0 * np.real(np.conj(phased_sum) * delta_phased_sum) + delta_population) / 20.0
        return float(delta_fidelity)

    def _liouvillian(self, ctrl_x: float, ctrl_y: float) -> np.ndarray:
        return self.l_d + ctrl_x * self.l_x + ctrl_y * self.l_y

    def _effective_generator(self, ctrl_x: float, ctrl_y: float) -> np.ndarray:
        return self.g_d + ctrl_x * self.g_x + ctrl_y * self.g_y

    def _unpack(self, variables: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        ctrl_x = np.asarray(variables[: self.config.num_tslots], dtype=np.float64)
        ctrl_y = np.asarray(
            variables[self.config.num_tslots : 2 * self.config.num_tslots],
            dtype=np.float64,
        )
        theta = float(np.mod(variables[-1], 2.0 * np.pi))
        return ctrl_x, ctrl_y, theta

    def _jitter_controls(self, base: np.ndarray, restart: int, axis: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed + 1000 * restart + axis)
        jitter = 0.08 * self.amp_bound * rng.normal(size=base.shape[0])
        return np.clip(base + jitter, -self.amp_bound, self.amp_bound)

    def _control_smoothness_cost(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> float:
        return self._smoothness_cost(ctrl_x) + self._smoothness_cost(ctrl_y)

    def _control_curvature_cost(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> float:
        return self._curvature_cost(ctrl_x) + self._curvature_cost(ctrl_y)

    @staticmethod
    def _smoothness_cost(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        delta = values[1:] - values[:-1]
        return float(np.mean(delta**2))

    @staticmethod
    def _smoothness_gradient(values: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(values)
        if len(values) < 2:
            return grad
        scale = 2.0 / (len(values) - 1)
        delta = values[1:] - values[:-1]
        grad[0] = -delta[0] * scale
        grad[-1] = delta[-1] * scale
        if len(values) > 2:
            grad[1:-1] = (delta[:-1] - delta[1:]) * scale
        return grad

    @staticmethod
    def _curvature_cost(values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        curvature = values[2:] - 2.0 * values[1:-1] + values[:-2]
        return float(np.mean(curvature**2))

    @staticmethod
    def _curvature_gradient(values: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(values)
        if len(values) < 3:
            return grad
        curvature = values[2:] - 2.0 * values[1:-1] + values[:-2]
        scale = 2.0 / curvature.size
        grad[:-2] += curvature * scale
        grad[1:-1] += -2.0 * curvature * scale
        grad[2:] += curvature * scale
        return grad

    def _vec_to_matrix(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.complex128).reshape((self.dimension, self.dimension), order="F")

    @staticmethod
    def _is_better(left: OpenSystemGRAPEResult, right: OpenSystemGRAPEResult) -> bool:
        if left.probe_fidelity > right.probe_fidelity + 1e-8:
            return True
        if abs(left.probe_fidelity - right.probe_fidelity) <= 1e-8 and left.objective < right.objective:
            return True
        return False
