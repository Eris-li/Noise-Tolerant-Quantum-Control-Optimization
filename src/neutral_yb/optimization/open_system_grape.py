from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import qutip
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize, minimize_scalar


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
    amplitude_diff_weight: float = 0.0
    phase_diff_weight: float = 0.0
    amplitude_diff_threshold: float = 0.01
    phase_diff_threshold: float = 0.1
    target_theta: float = 0.0
    fidelity_target: float = 0.995
    objective_metric: str = "special_state"
    benchmark_active_channel: bool = False
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
    objective_fidelity: float
    probe_fidelity: float
    active_channel_fidelity: float | None
    objective_metric: str
    objective: float
    num_iter: int
    num_fid_func_calls: int
    wall_time: float
    termination_reason: str
    evo_time: float
    num_tslots: int
    control_smoothness_cost: float
    control_curvature_cost: float
    amplitude_diff_cost: float
    phase_diff_cost: float
    success: bool

    def to_json(self) -> dict[str, float | int | str | bool | list[float] | None]:
        return {
            "ctrl_x": [float(x) for x in self.ctrl_x],
            "ctrl_y": [float(x) for x in self.ctrl_y],
            "amplitudes": [float(x) for x in self.amplitudes],
            "phases": [float(x) for x in self.phases],
            "target_theta": float(self.target_theta),
            "optimized_theta": float(self.optimized_theta),
            "fid_err": float(self.fid_err),
            "objective_fidelity": float(self.objective_fidelity),
            "probe_fidelity": float(self.probe_fidelity),
            "active_channel_fidelity": None
            if self.active_channel_fidelity is None
            else float(self.active_channel_fidelity),
            "phase_gate_fidelity": float(self.probe_fidelity),
            "fidelity_metric": "paper_eq7_special_state_phase_gate_fidelity",
            "objective_metric": self.objective_metric,
            "objective": float(self.objective),
            "num_iter": int(self.num_iter),
            "num_fid_func_calls": int(self.num_fid_func_calls),
            "wall_time": float(self.wall_time),
            "termination_reason": self.termination_reason,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "control_smoothness_cost": float(self.control_smoothness_cost),
            "control_curvature_cost": float(self.control_curvature_cost),
            "amplitude_diff_cost": float(self.amplitude_diff_cost),
            "phase_diff_cost": float(self.phase_diff_cost),
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
    r"""Open-system GRAPE for the active {|01>, |11>} manifold.

    The default optimization target follows arXiv:2202.00903. It propagates
    the unnormalized state

        |psi(0)> = |01> + |11>

    and evaluates

        F = (|1 + 2 a01 + a11|^2 + 1 + 2|a01|^2 + |a11|^2) / 20

    with

        a01 = exp(-i theta) <01|psi(T)>
        a11 = -exp(-2 i theta) <11|psi(T)>

    For the open-system model, the optimization propagates this state under
    the effective non-Hermitian generator

        G = -i H - 1/2 sum_k C_k^\dagger C_k

    while keeping exact Liouvillian helpers available for diagnostics. Setting
    ``OpenSystemGRAPEConfig.objective_metric="active_channel"`` switches the
    target to the exact process fidelity of the reduced active-subspace
    superoperator. When an ensemble of models is provided, the optimization
    target is the arithmetic mean over the quasistatic ensemble.
    """

    def __init__(self, model, config: OpenSystemGRAPEConfig, ensemble_models: list | None = None):
        self.model = model
        self.config = config
        self.dimension = int(model.dimension())
        self.vector_dimension = self.dimension * self.dimension
        self.active_indices = tuple(int(index) for index in model.active_gate_indices())
        self.active_dim = len(self.active_indices)
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
        self.phase_identity = np.eye(self.dimension, dtype=np.complex128)
        self.liou_identity = np.eye(self.vector_dimension, dtype=np.complex128)
        self.ensemble_models = [self.model] if ensemble_models is None else list(ensemble_models)
        self.ensemble_data = [self._build_member_data(member) for member in self.ensemble_models]
        for member in self.ensemble_models:
            if tuple(int(index) for index in member.active_gate_indices()) != self.active_indices:
                raise ValueError("All ensemble models must use the same active gate indices")
            if int(member.dimension()) != self.dimension:
                raise ValueError("All ensemble models must have the same Hilbert-space dimension")
            if abs(float(member.control_amplitude_bound()) - self.amp_bound) > 1e-9:
                raise ValueError("All ensemble models must share the same control amplitude bound")
        self.initial_phase_state = np.asarray(
            model.special_phase_gate_state().full(),
            dtype=np.complex128,
        ).ravel()
        self.active_operator_basis = self._build_active_operator_basis()
        self.active_reducer = self._build_active_reducer()

    def reconfigure(self, config: OpenSystemGRAPEConfig) -> None:
        self.config = config

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
        best = self._zero_control_baseline_result()
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
                    f"F={candidate.objective_fidelity:.6f} obj={candidate.objective:.6e} "
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
            self.reconfigure(
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
                    amplitude_diff_weight=self.config.amplitude_diff_weight,
                    phase_diff_weight=self.config.phase_diff_weight,
                    amplitude_diff_threshold=self.config.amplitude_diff_threshold,
                    phase_diff_threshold=self.config.phase_diff_threshold,
                    target_theta=self.config.target_theta,
                    fidelity_target=self.config.fidelity_target,
                    objective_metric=self.config.objective_metric,
                    benchmark_active_channel=self.config.benchmark_active_channel,
                    show_progress=self.config.show_progress,
                )
            )
            if self.config.show_progress:
                print(
                    f"[scan] starting {index}/{len(durations)} "
                    f"T={duration:.3f} slots={self.config.num_tslots}",
                    flush=True,
                )
            result = self.optimize(
                initial_ctrl_x=ctrl_x,
                initial_ctrl_y=ctrl_y,
                initial_theta=theta,
            )
            results.append(result)
            fidelities.append(result.objective_fidelity)
            ctrl_x, ctrl_y, theta = result.ctrl_x, result.ctrl_y, result.optimized_theta
            if self.config.show_progress:
                step_elapsed = time.perf_counter() - started_at
                total_elapsed = time.perf_counter() - scan_started_at
                avg_per_step = total_elapsed / index
                remaining = avg_per_step * (len(durations) - index)
                print(
                    f"[scan] {index}/{len(durations)} ({100.0*index/len(durations):5.1f}%) "
                    f"T={duration:.3f} F={result.objective_fidelity:.6f} "
                    f"step={step_elapsed:7.1f}s elapsed={total_elapsed:7.1f}s "
                    f"remaining_T={len(durations)-index:2d} eta={remaining:7.1f}s",
                    flush=True,
                )

        qualified = [res for res in results if res.objective_fidelity >= self.config.fidelity_target]
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
        final_states: list[qutip.Qobj] = []
        for source_ket, _target in self.model.probe_kets(theta=0.0):
            final_states.append(self.evolve_density_matrix(ctrl_x, ctrl_y, qutip.ket2dm(source_ket)))
        return final_states

    def evolve_density_matrix(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        rho0: qutip.Qobj,
    ) -> qutip.Qobj:
        vector = np.asarray(rho0.full(), dtype=np.complex128).reshape(-1, order="F")
        member = self.ensemble_data[0]
        vector = member["liou_prefix"] @ vector
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            vector = expm(self.config.dt * self._liouvillian(float(x_value), float(y_value))) @ vector
        vector = member["liou_suffix"] @ vector
        return qutip.Qobj(self._vec_to_matrix(vector))

    def trajectory(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        rho0: qutip.Qobj,
    ) -> tuple[np.ndarray, list[qutip.Qobj]]:
        vector = np.asarray(rho0.full(), dtype=np.complex128).reshape(-1, order="F")
        member = self.ensemble_data[0]
        states = [qutip.Qobj(self._vec_to_matrix(vector))]
        times = [0.0]
        current_time = 0.0
        trajectory_cache = self._build_fixed_clock_trajectory_segments(self.ensemble_models[0], member["l_d"])
        for propagator, dt in zip(trajectory_cache["liou_prefix_steps"], trajectory_cache["liou_prefix_dts"]):
            vector = propagator @ vector
            current_time += dt
            times.append(current_time)
            states.append(qutip.Qobj(self._vec_to_matrix(vector)))
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            vector = expm(self.config.dt * self._liouvillian(float(x_value), float(y_value))) @ vector
            current_time += self.config.dt
            times.append(current_time)
            states.append(qutip.Qobj(self._vec_to_matrix(vector)))
        for propagator, dt in zip(trajectory_cache["liou_suffix_steps"], trajectory_cache["liou_suffix_dts"]):
            vector = propagator @ vector
            current_time += dt
            times.append(current_time)
            states.append(qutip.Qobj(self._vec_to_matrix(vector)))
        return np.asarray(times, dtype=np.float64), states

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
        metric = self._resolve_objective_metric()
        if metric == "special_state":
            return self._objective_and_gradient_special_state(variables)
        return self._objective_and_gradient_active_channel(variables)

    def _objective_and_gradient_special_state(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        ctrl_x, ctrl_y, theta = self._unpack(variables)
        dt = self.config.dt
        fidelity = 0.0
        fidelity_theta_grad = 0.0
        ctrl_x_grad = np.zeros_like(ctrl_x)
        ctrl_y_grad = np.zeros_like(ctrl_y)

        for member in self.ensemble_data:
            slice_propags: list[np.ndarray] = []
            current_state = member["phase_prefix"] @ self.initial_phase_state
            state_prefix: list[np.ndarray] = [current_state]

            for x_value, y_value in zip(ctrl_x, ctrl_y):
                g_k = member["g_d"] + float(x_value) * member["g_x"] + float(y_value) * member["g_y"]
                u_k = expm(dt * g_k)
                slice_propags.append(u_k)
                current_state = u_k @ current_state
                state_prefix.append(current_state)

            final_state = member["phase_suffix"] @ state_prefix[-1]
            member_fidelity, member_theta_grad = self._phase_gate_fidelity_and_theta_gradient(final_state, theta)
            fidelity += member_fidelity
            fidelity_theta_grad += member_theta_grad

            suffix_propags: list[np.ndarray] = [self.phase_identity for _ in range(self.config.num_tslots)]
            current_suffix = member["phase_suffix"]
            for index in range(self.config.num_tslots - 1, -1, -1):
                suffix_propags[index] = current_suffix
                current_suffix = current_suffix @ slice_propags[index]

            for index, (x_value, y_value) in enumerate(zip(ctrl_x, ctrl_y)):
                g_k = member["g_d"] + float(x_value) * member["g_x"] + float(y_value) * member["g_y"]
                du_x = expm_frechet(dt * g_k, dt * member["g_x"], compute_expm=False)
                du_y = expm_frechet(dt * g_k, dt * member["g_y"], compute_expm=False)
                d_state_x = suffix_propags[index] @ du_x @ state_prefix[index]
                d_state_y = suffix_propags[index] @ du_y @ state_prefix[index]
                ctrl_x_grad[index] -= self._phase_gate_fidelity_state_gradient(final_state, d_state_x, theta)
                ctrl_y_grad[index] -= self._phase_gate_fidelity_state_gradient(final_state, d_state_y, theta)

        ensemble_size = float(len(self.ensemble_data))
        fidelity /= ensemble_size
        fidelity_theta_grad /= ensemble_size
        ctrl_x_grad /= ensemble_size
        ctrl_y_grad /= ensemble_size
        smoothness_cost = self._control_smoothness_cost(ctrl_x, ctrl_y)
        curvature_cost = self._control_curvature_cost(ctrl_x, ctrl_y)
        amplitude_diff_cost, phase_diff_cost, amp_penalty_x, amp_penalty_y, phase_penalty_x, phase_penalty_y = (
            self._amplitude_phase_diff_penalty(ctrl_x, ctrl_y)
        )
        objective = (
            1.0 - fidelity
            + self.config.control_smoothness_weight * smoothness_cost
            + self.config.control_curvature_weight * curvature_cost
            + self.config.amplitude_diff_weight * amplitude_diff_cost
            + self.config.phase_diff_weight * phase_diff_cost
        )

        if self.config.control_smoothness_weight > 0.0:
            ctrl_x_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_y)
        if self.config.control_curvature_weight > 0.0:
            ctrl_x_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_y)
        if self.config.amplitude_diff_weight > 0.0:
            ctrl_x_grad += self.config.amplitude_diff_weight * amp_penalty_x
            ctrl_y_grad += self.config.amplitude_diff_weight * amp_penalty_y
        if self.config.phase_diff_weight > 0.0:
            ctrl_x_grad += self.config.phase_diff_weight * phase_penalty_x
            ctrl_y_grad += self.config.phase_diff_weight * phase_penalty_y

        gradient = np.concatenate([ctrl_x_grad, ctrl_y_grad, np.array([-fidelity_theta_grad])])
        return float(objective), gradient

    def _objective_and_gradient_active_channel(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        ctrl_x, ctrl_y, theta = self._unpack(variables)
        dt = self.config.dt
        fidelity = 0.0
        fidelity_theta_grad = 0.0
        ctrl_x_grad = np.zeros_like(ctrl_x)
        ctrl_y_grad = np.zeros_like(ctrl_y)

        for member in self.ensemble_data:
            slice_propags: list[np.ndarray] = []
            current_batch = member["liou_prefix"] @ self.active_operator_basis
            batch_prefix: list[np.ndarray] = [current_batch]

            for x_value, y_value in zip(ctrl_x, ctrl_y):
                l_k = member["l_d"] + float(x_value) * member["l_x"] + float(y_value) * member["l_y"]
                u_k = expm(dt * l_k)
                slice_propags.append(u_k)
                current_batch = u_k @ current_batch
                batch_prefix.append(current_batch)

            final_batch = member["liou_suffix"] @ batch_prefix[-1]
            member_fidelity, member_theta_grad = self._channel_fidelity_and_theta_gradient(final_batch, theta)
            fidelity += member_fidelity
            fidelity_theta_grad += member_theta_grad

            suffix_propags: list[np.ndarray] = [self.liou_identity for _ in range(self.config.num_tslots)]
            current_suffix = member["liou_suffix"]
            for index in range(self.config.num_tslots - 1, -1, -1):
                suffix_propags[index] = current_suffix
                current_suffix = current_suffix @ slice_propags[index]

            for index, (x_value, y_value) in enumerate(zip(ctrl_x, ctrl_y)):
                l_k = member["l_d"] + float(x_value) * member["l_x"] + float(y_value) * member["l_y"]
                du_x = expm_frechet(dt * l_k, dt * member["l_x"], compute_expm=False)
                du_y = expm_frechet(dt * l_k, dt * member["l_y"], compute_expm=False)
                d_batch_x = suffix_propags[index] @ du_x @ batch_prefix[index]
                d_batch_y = suffix_propags[index] @ du_y @ batch_prefix[index]
                ctrl_x_grad[index] -= self._channel_fidelity_batch_gradient(d_batch_x, theta)
                ctrl_y_grad[index] -= self._channel_fidelity_batch_gradient(d_batch_y, theta)

        ensemble_size = float(len(self.ensemble_data))
        fidelity /= ensemble_size
        fidelity_theta_grad /= ensemble_size
        ctrl_x_grad /= ensemble_size
        ctrl_y_grad /= ensemble_size
        smoothness_cost = self._control_smoothness_cost(ctrl_x, ctrl_y)
        curvature_cost = self._control_curvature_cost(ctrl_x, ctrl_y)
        amplitude_diff_cost, phase_diff_cost, amp_penalty_x, amp_penalty_y, phase_penalty_x, phase_penalty_y = (
            self._amplitude_phase_diff_penalty(ctrl_x, ctrl_y)
        )
        objective = (
            1.0 - fidelity
            + self.config.control_smoothness_weight * smoothness_cost
            + self.config.control_curvature_weight * curvature_cost
            + self.config.amplitude_diff_weight * amplitude_diff_cost
            + self.config.phase_diff_weight * phase_diff_cost
        )

        if self.config.control_smoothness_weight > 0.0:
            ctrl_x_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_smoothness_weight * self._smoothness_gradient(ctrl_y)
        if self.config.control_curvature_weight > 0.0:
            ctrl_x_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_x)
            ctrl_y_grad += self.config.control_curvature_weight * self._curvature_gradient(ctrl_y)
        if self.config.amplitude_diff_weight > 0.0:
            ctrl_x_grad += self.config.amplitude_diff_weight * amp_penalty_x
            ctrl_y_grad += self.config.amplitude_diff_weight * amp_penalty_y
        if self.config.phase_diff_weight > 0.0:
            ctrl_x_grad += self.config.phase_diff_weight * phase_penalty_x
            ctrl_y_grad += self.config.phase_diff_weight * phase_penalty_y

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
        probe_fidelity = self.phase_gate_fidelity(ctrl_x, ctrl_y, theta)
        active_channel_fidelity: float | None = None
        if self._resolve_objective_metric() == "active_channel" or self.config.benchmark_active_channel:
            active_channel_fidelity = self.channel_fidelity(ctrl_x, ctrl_y, theta)
        objective_fidelity = (
            probe_fidelity
            if self._resolve_objective_metric() == "special_state"
            else float(active_channel_fidelity)
        )
        objective = (
            1.0
            - objective_fidelity
            + self.config.control_smoothness_weight * self._control_smoothness_cost(ctrl_x, ctrl_y)
            + self.config.control_curvature_weight * self._control_curvature_cost(ctrl_x, ctrl_y)
            + self.config.amplitude_diff_weight * self._amplitude_phase_diff_penalty(ctrl_x, ctrl_y)[0]
            + self.config.phase_diff_weight * self._amplitude_phase_diff_penalty(ctrl_x, ctrl_y)[1]
        )
        amplitude_diff_cost, phase_diff_cost, _, _, _, _ = self._amplitude_phase_diff_penalty(ctrl_x, ctrl_y)
        return OpenSystemGRAPEResult(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            amplitudes=amplitudes,
            phases=phases,
            target_theta=float(self.config.target_theta),
            optimized_theta=float(theta),
            fid_err=float(1.0 - objective_fidelity),
            objective_fidelity=float(objective_fidelity),
            probe_fidelity=float(probe_fidelity),
            active_channel_fidelity=None if active_channel_fidelity is None else float(active_channel_fidelity),
            objective_metric=self._objective_metric_label(),
            objective=float(objective),
            num_iter=int(num_iter),
            num_fid_func_calls=int(num_fid_func_calls),
            wall_time=float(wall_time),
            termination_reason=termination_reason,
            evo_time=float(self.config.evo_time),
            num_tslots=int(self.config.num_tslots),
            control_smoothness_cost=float(self._control_smoothness_cost(ctrl_x, ctrl_y)),
            control_curvature_cost=float(self._control_curvature_cost(ctrl_x, ctrl_y)),
            amplitude_diff_cost=float(amplitude_diff_cost),
            phase_diff_cost=float(phase_diff_cost),
            success=bool(success),
        )

    def _zero_control_baseline_result(self) -> OpenSystemGRAPEResult:
        ctrl_x = np.zeros(self.config.num_tslots, dtype=np.float64)
        ctrl_y = np.zeros(self.config.num_tslots, dtype=np.float64)
        if self._resolve_objective_metric() == "special_state":
            theta, objective_fidelity = self.optimize_theta_for_phase_fidelity(ctrl_x, ctrl_y)
        else:
            theta, objective_fidelity = self.optimize_theta_for_channel(ctrl_x, ctrl_y)
        probe_fidelity = self.phase_gate_fidelity(ctrl_x, ctrl_y, theta)
        active_channel_fidelity: float | None = None
        if self._resolve_objective_metric() == "active_channel" or self.config.benchmark_active_channel:
            active_channel_fidelity = self.channel_fidelity(ctrl_x, ctrl_y, theta)
        amplitudes, phases = self.model.control_cartesian_to_polar(ctrl_x, ctrl_y)
        return OpenSystemGRAPEResult(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            amplitudes=amplitudes,
            phases=phases,
            target_theta=float(self.config.target_theta),
            optimized_theta=float(theta),
            fid_err=float(1.0 - objective_fidelity),
            objective_fidelity=float(objective_fidelity),
            probe_fidelity=float(probe_fidelity),
            active_channel_fidelity=None if active_channel_fidelity is None else float(active_channel_fidelity),
            objective_metric=self._objective_metric_label(),
            objective=float(1.0 - objective_fidelity),
            num_iter=0,
            num_fid_func_calls=1,
            wall_time=0.0,
            termination_reason="zero-control baseline",
            evo_time=float(self.config.evo_time),
            num_tslots=int(self.config.num_tslots),
            control_smoothness_cost=0.0,
            control_curvature_cost=0.0,
            amplitude_diff_cost=0.0,
            phase_diff_cost=0.0,
            success=True,
        )

    def final_phase_state(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> np.ndarray:
        return self.final_phase_state_member(ctrl_x, ctrl_y, member_index=0)

    def final_phase_state_member(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, member_index: int = 0) -> np.ndarray:
        member = self.ensemble_data[member_index]
        current_state = member["phase_prefix"] @ self.initial_phase_state
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            g_k = member["g_d"] + float(x_value) * member["g_x"] + float(y_value) * member["g_y"]
            current_state = expm(self.config.dt * g_k) @ current_state
        return member["phase_suffix"] @ current_state

    def phase_gate_fidelity(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, theta: float) -> float:
        values = [
            self._phase_gate_fidelity_and_theta_gradient(
                self.final_phase_state_member(ctrl_x, ctrl_y, member_index=index),
                theta,
            )[0]
            for index in range(len(self.ensemble_data))
        ]
        return float(np.mean(values))

    def optimize_theta_for_phase_fidelity(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> tuple[float, float]:
        result = minimize_scalar(
            lambda theta: -self.phase_gate_fidelity(ctrl_x, ctrl_y, float(theta)),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.phase_gate_fidelity(ctrl_x, ctrl_y, theta)

    def channel_superoperator(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, member_index: int = 0) -> np.ndarray:
        member = self.ensemble_data[member_index]
        current_batch = member["liou_prefix"] @ self.active_operator_basis
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            l_k = member["l_d"] + float(x_value) * member["l_x"] + float(y_value) * member["l_y"]
            current_batch = expm(self.config.dt * l_k) @ current_batch
        return self.active_reducer @ (member["liou_suffix"] @ current_batch)

    def channel_fidelity(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, theta: float) -> float:
        values = [
            self._channel_fidelity_from_superoperator(self.channel_superoperator(ctrl_x, ctrl_y, member_index=index), theta)
            for index in range(len(self.ensemble_data))
        ]
        return float(np.mean(values))

    def optimize_theta_for_channel(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> tuple[float, float]:
        result = minimize_scalar(
            lambda theta: -self.channel_fidelity(ctrl_x, ctrl_y, float(theta)),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.channel_fidelity(ctrl_x, ctrl_y, theta)

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

    def _build_member_data(self, member) -> dict[str, np.ndarray]:
        h_d = np.asarray(member.drift_hamiltonian().full(), dtype=np.complex128)
        h_x, h_y = [np.asarray(operator.full(), dtype=np.complex128) for operator in member.lower_leg_control_hamiltonians()]
        decay_matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for operator in member.collapse_operators():
            c_matrix = np.asarray(operator.full(), dtype=np.complex128)
            decay_matrix += c_matrix.conj().T @ c_matrix
        l_x, l_y = [np.asarray(operator.full(), dtype=np.complex128) for operator in member.control_liouvillians()]
        data = {
            "g_d": -1j * h_d - 0.5 * decay_matrix,
            "g_x": -1j * h_x,
            "g_y": -1j * h_y,
            "l_d": np.asarray(member.drift_liouvillian().full(), dtype=np.complex128),
            "l_x": l_x,
            "l_y": l_y,
        }
        clock_segments = self._build_fixed_clock_segments(member, data["g_d"], data["l_d"])
        data.update(clock_segments)
        return data

    def _build_fixed_clock_segments(
        self,
        member,
        g_d: np.ndarray,
        l_d: np.ndarray,
    ) -> dict[str, object]:
        if hasattr(member, "fixed_clock_segment_cache"):
            cache = member.fixed_clock_segment_cache
            return {
                "phase_prefix": np.asarray(cache["phase_prefix"], dtype=np.complex128),
                "phase_suffix": np.asarray(cache["phase_suffix"], dtype=np.complex128),
                "liou_prefix": np.asarray(cache["liou_prefix"], dtype=np.complex128),
                "liou_suffix": np.asarray(cache["liou_suffix"], dtype=np.complex128),
            }
        if not hasattr(member, "clock_control_hamiltonians") or not hasattr(member, "clock_segment_controls"):
            return {
                "phase_prefix": self.phase_identity,
                "phase_suffix": self.phase_identity,
                "liou_prefix": self.liou_identity,
                "liou_suffix": self.liou_identity,
            }

        h_clock_x, h_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128)
            for operator in member.clock_control_hamiltonians()
        ]
        g_clock_x = -1j * h_clock_x
        g_clock_y = -1j * h_clock_y
        l_clock_x, l_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128)
            for operator in member.clock_control_liouvillians()
        ]
        segments = member.clock_segment_controls()

        def build_phase_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(float(dt) * (g_d + float(x_value) * g_clock_x + float(y_value) * g_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        def build_liou_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(float(dt) * (l_d + float(x_value) * l_clock_x + float(y_value) * l_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        prefix_phase_steps = build_phase_steps(
            np.asarray(segments["prefix_x"], dtype=np.float64),
            np.asarray(segments["prefix_y"], dtype=np.float64),
            float(segments["prefix_dt"]),
        )
        suffix_phase_steps = build_phase_steps(
            np.asarray(segments["suffix_x"], dtype=np.float64),
            np.asarray(segments["suffix_y"], dtype=np.float64),
            float(segments["suffix_dt"]),
        )
        prefix_liou_steps = build_liou_steps(
            np.asarray(segments["prefix_x"], dtype=np.float64),
            np.asarray(segments["prefix_y"], dtype=np.float64),
            float(segments["prefix_dt"]),
        )
        suffix_liou_steps = build_liou_steps(
            np.asarray(segments["suffix_x"], dtype=np.float64),
            np.asarray(segments["suffix_y"], dtype=np.float64),
            float(segments["suffix_dt"]),
        )
        return {
            "phase_prefix": self._compose_propagators(prefix_phase_steps, self.phase_identity),
            "phase_suffix": self._compose_propagators(suffix_phase_steps, self.phase_identity),
            "liou_prefix": self._compose_propagators(prefix_liou_steps, self.liou_identity),
            "liou_suffix": self._compose_propagators(suffix_liou_steps, self.liou_identity),
        }

    def _build_fixed_clock_trajectory_segments(
        self,
        member,
        l_d: np.ndarray,
    ) -> dict[str, object]:
        if hasattr(member, "fixed_clock_trajectory_cache"):
            cache = member.fixed_clock_trajectory_cache
            return {
                "liou_prefix_steps": [np.asarray(step, dtype=np.complex128) for step in cache["liou_prefix_steps"]],
                "liou_prefix_dts": [float(value) for value in cache["liou_prefix_dts"]],
                "liou_suffix_steps": [np.asarray(step, dtype=np.complex128) for step in cache["liou_suffix_steps"]],
                "liou_suffix_dts": [float(value) for value in cache["liou_suffix_dts"]],
            }
        if not hasattr(member, "clock_control_hamiltonians") or not hasattr(member, "clock_segment_controls"):
            return {
                "liou_prefix_steps": [],
                "liou_prefix_dts": [],
                "liou_suffix_steps": [],
                "liou_suffix_dts": [],
            }

        l_clock_x, l_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128)
            for operator in member.clock_control_liouvillians()
        ]
        segments = member.clock_segment_controls()

        def build_liou_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(float(dt) * (l_d + float(x_value) * l_clock_x + float(y_value) * l_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        prefix_liou_steps = build_liou_steps(
            np.asarray(segments["prefix_x"], dtype=np.float64),
            np.asarray(segments["prefix_y"], dtype=np.float64),
            float(segments["prefix_dt"]),
        )
        suffix_liou_steps = build_liou_steps(
            np.asarray(segments["suffix_x"], dtype=np.float64),
            np.asarray(segments["suffix_y"], dtype=np.float64),
            float(segments["suffix_dt"]),
        )
        return {
            "liou_prefix_steps": prefix_liou_steps,
            "liou_prefix_dts": [float(segments["prefix_dt"])] * len(prefix_liou_steps),
            "liou_suffix_steps": suffix_liou_steps,
            "liou_suffix_dts": [float(segments["suffix_dt"])] * len(suffix_liou_steps),
        }

    @staticmethod
    def _compose_propagators(propagators: list[np.ndarray], identity: np.ndarray) -> np.ndarray:
        total = np.array(identity, copy=True)
        for propagator in propagators:
            total = propagator @ total
        return total

    def _build_active_operator_basis(self) -> np.ndarray:
        basis = np.zeros((self.vector_dimension, self.active_dim * self.active_dim), dtype=np.complex128)
        column = 0
        for active_col in self.active_indices:
            for active_row in self.active_indices:
                operator = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
                operator[active_row, active_col] = 1.0
                basis[:, column] = operator.reshape(-1, order="F")
                column += 1
        return basis

    def _build_active_reducer(self) -> np.ndarray:
        reducer = np.zeros((self.active_dim * self.active_dim, self.vector_dimension), dtype=np.complex128)
        column = 0
        for active_col in self.active_indices:
            for active_row in self.active_indices:
                reducer[column, active_row + active_col * self.dimension] = 1.0
                column += 1
        return reducer

    def _target_channel_superoperator(self, theta: float) -> np.ndarray:
        unitary = np.diag([np.exp(1j * theta), -np.exp(2j * theta)])
        return np.kron(np.conjugate(unitary), unitary)

    def _target_channel_superoperator_derivative(self, theta: float) -> np.ndarray:
        unitary = np.diag([np.exp(1j * theta), -np.exp(2j * theta)])
        d_unitary = np.diag([1j * np.exp(1j * theta), -2j * np.exp(2j * theta)])
        return np.kron(np.conjugate(d_unitary), unitary) + np.kron(np.conjugate(unitary), d_unitary)

    def _channel_fidelity_from_superoperator(self, superoperator: np.ndarray, theta: float) -> float:
        target = self._target_channel_superoperator(theta)
        return float(np.real(np.vdot(target, superoperator)) / float(self.active_dim * self.active_dim))

    def _channel_fidelity_and_theta_gradient(self, final_batch: np.ndarray, theta: float) -> tuple[float, float]:
        superoperator = (
            self.active_reducer @ final_batch
            if final_batch.shape[0] == self.vector_dimension
            else np.asarray(final_batch, dtype=np.complex128)
        )
        target = self._target_channel_superoperator(theta)
        target_derivative = self._target_channel_superoperator_derivative(theta)
        normalization = float(self.active_dim * self.active_dim)
        fidelity = np.real(np.vdot(target, superoperator)) / normalization
        theta_gradient = np.real(np.vdot(target_derivative, superoperator)) / normalization
        return float(fidelity), float(theta_gradient)

    def _channel_fidelity_batch_gradient(self, d_batch: np.ndarray, theta: float) -> float:
        target = self._target_channel_superoperator(theta)
        delta_superoperator = self.active_reducer @ d_batch
        normalization = float(self.active_dim * self.active_dim)
        return float(np.real(np.vdot(target, delta_superoperator)) / normalization)

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

    def _amplitude_phase_diff_penalty(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        amp_grad_x = np.zeros_like(ctrl_x)
        amp_grad_y = np.zeros_like(ctrl_y)
        phase_grad_x = np.zeros_like(ctrl_x)
        phase_grad_y = np.zeros_like(ctrl_y)
        if len(ctrl_x) < 2:
            return 0.0, 0.0, amp_grad_x, amp_grad_y, phase_grad_x, phase_grad_y

        radius = np.sqrt(ctrl_x**2 + ctrl_y**2)
        amp_fraction = radius / self.amp_bound
        delta_amp = amp_fraction[1:] - amp_fraction[:-1]
        amp_abs = np.abs(delta_amp)
        amp_excess = np.maximum(amp_abs - self.config.amplitude_diff_threshold, 0.0)
        amp_cost = float(np.mean(amp_excess**2))

        if self.config.amplitude_diff_weight > 0.0 and amp_excess.size > 0:
            scale = 2.0 / amp_excess.size
            active = amp_excess > 0.0
            amp_delta_grad = np.zeros_like(delta_amp)
            amp_delta_grad[active] = scale * amp_excess[active] * np.sign(delta_amp[active])
            safe_radius = np.where(radius > 1e-12, radius, np.inf)
            d_amp_dx = np.where(radius > 1e-12, ctrl_x / (self.amp_bound * safe_radius), 0.0)
            d_amp_dy = np.where(radius > 1e-12, ctrl_y / (self.amp_bound * safe_radius), 0.0)
            amp_grad_x[:-1] -= amp_delta_grad * d_amp_dx[:-1]
            amp_grad_x[1:] += amp_delta_grad * d_amp_dx[1:]
            amp_grad_y[:-1] -= amp_delta_grad * d_amp_dy[:-1]
            amp_grad_y[1:] += amp_delta_grad * d_amp_dy[1:]

        phases = np.arctan2(ctrl_y, ctrl_x)
        delta_phase = (phases[1:] - phases[:-1] + np.pi) % (2.0 * np.pi) - np.pi
        phase_abs = np.abs(delta_phase)
        phase_excess = np.maximum(phase_abs - self.config.phase_diff_threshold, 0.0)
        phase_cost = float(np.mean(phase_excess**2))

        if self.config.phase_diff_weight > 0.0 and phase_excess.size > 0:
            scale = 2.0 / phase_excess.size
            active = phase_excess > 0.0
            phase_delta_grad = np.zeros_like(delta_phase)
            phase_delta_grad[active] = scale * phase_excess[active] * np.sign(delta_phase[active])
            radius_sq = ctrl_x**2 + ctrl_y**2
            safe_radius_sq = np.where(radius_sq > 1e-12, radius_sq, np.inf)
            d_phase_dx = np.where(radius_sq > 1e-12, -ctrl_y / safe_radius_sq, 0.0)
            d_phase_dy = np.where(radius_sq > 1e-12, ctrl_x / safe_radius_sq, 0.0)
            phase_grad_x[:-1] -= phase_delta_grad * d_phase_dx[:-1]
            phase_grad_x[1:] += phase_delta_grad * d_phase_dx[1:]
            phase_grad_y[:-1] -= phase_delta_grad * d_phase_dy[:-1]
            phase_grad_y[1:] += phase_delta_grad * d_phase_dy[1:]

        return amp_cost, phase_cost, amp_grad_x, amp_grad_y, phase_grad_x, phase_grad_y

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

    def _resolve_objective_metric(self) -> str:
        metric = str(self.config.objective_metric).strip().lower()
        if metric not in {"special_state", "active_channel"}:
            raise ValueError(
                "OpenSystemGRAPEConfig.objective_metric must be 'special_state' or 'active_channel'"
            )
        return metric

    def _objective_metric_label(self) -> str:
        if self._resolve_objective_metric() == "special_state":
            return "paper_eq7_special_state_phase_gate_fidelity"
        return "active_subspace_process_fidelity"

    @staticmethod
    def _is_better(left: OpenSystemGRAPEResult, right: OpenSystemGRAPEResult) -> bool:
        if left.objective_fidelity > right.objective_fidelity + 1e-8:
            return True
        if abs(left.objective_fidelity - right.objective_fidelity) <= 1e-8 and left.objective < right.objective:
            return True
        return False
