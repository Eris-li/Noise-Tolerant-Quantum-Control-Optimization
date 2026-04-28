from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from neutral_yb.models.ma2023_pulse import (
    controls_from_envelope_phase,
    phase_regularization,
    validate_phase_only_pulse,
    wrap_phase,
)
from neutral_yb.models.ma2023_six_level import Ma2023PerfectBlockadeSixLevelModel


@dataclass(frozen=True)
class Ma2023SixLevelGRAPEConfig:
    num_tslots: int = 100
    evo_time: float = 12.3876
    max_iter: int = 160
    num_restarts: int = 4
    seed: int = 31
    phase_smoothness_weight: float = 1e-3
    phase_curvature_weight: float = 1e-3
    chebyshev_degree: int = 13
    chebyshev_init_scale: float = 1.0
    chebyshev_coefficient_bound: float | None = 20.0
    fidelity_target: float = 0.99999
    show_progress: bool = False

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class Ma2023SixLevelGRAPEResult:
    ctrl_x: np.ndarray
    ctrl_y: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    theta0: float
    theta1: float
    fidelity: float
    process_fidelity: float
    leakage: float
    objective: float
    num_iter: int
    num_fid_func_calls: int
    wall_time: float
    success: bool
    termination_reason: str
    parameterization: str = "direct_phase"
    phase_rate_coefficients: np.ndarray | None = None
    phase_origin: float | None = None

    def to_json(self) -> dict[str, object]:
        validation = validate_phase_only_pulse(self.amplitudes, self.phases)
        payload: dict[str, object] = {
            "ctrl_x": [float(value) for value in self.ctrl_x],
            "ctrl_y": [float(value) for value in self.ctrl_y],
            "amplitudes": [float(value) for value in self.amplitudes],
            "optimized_phase_rad_bounded": [float(value) for value in self.phases],
            "theta0": float(self.theta0),
            "theta1": float(self.theta1),
            "fidelity": float(self.fidelity),
            "process_fidelity": float(self.process_fidelity),
            "leakage": float(self.leakage),
            "objective": float(self.objective),
            "num_iter": int(self.num_iter),
            "num_fid_func_calls": int(self.num_fid_func_calls),
            "wall_time": float(self.wall_time),
            "success": bool(self.success),
            "termination_reason": self.termination_reason,
            "pulse_validation": validation,
            "fidelity_definition": "Ma2023 Methods Eq. 10 using a diagonal perfect-blockade channel",
            "phase_parameterization": self.parameterization,
        }
        if self.phase_rate_coefficients is not None:
            payload["phase_rate_chebyshev_coefficients"] = [
                float(value) for value in self.phase_rate_coefficients
            ]
        if self.phase_origin is not None:
            payload["phase_origin"] = float(self.phase_origin)
        return payload


class Ma2023SixLevelPhaseOptimizer:
    """Phase-only GRAPE optimizer for the Ma 2023 perfect-blockade six-level model."""

    def __init__(
        self,
        *,
        model: Ma2023PerfectBlockadeSixLevelModel,
        config: Ma2023SixLevelGRAPEConfig,
        envelope: np.ndarray,
    ) -> None:
        self.model = model
        self.config = config
        self.envelope = np.asarray(envelope, dtype=np.float64)
        self.dimension = int(model.dimension())
        self.h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
        h_x, h_y = model.lower_leg_control_hamiltonians()
        self.g_d = -1j * self.h_d - 0.5 * self._decay_matrix()
        self.g_x = -1j * np.asarray(h_x.full(), dtype=np.complex128)
        self.g_y = -1j * np.asarray(h_y.full(), dtype=np.complex128)
        self.initial_indices = model.computational_indices()

    def optimize(self) -> Ma2023SixLevelGRAPEResult:
        rng = np.random.default_rng(self.config.seed)
        slots = int(self.envelope.size)
        best: Ma2023SixLevelGRAPEResult | None = None
        started_at = time.perf_counter()

        for restart in range(self.config.num_restarts):
            grid = np.linspace(0.0, 2.0 * np.pi, slots, endpoint=False, dtype=np.float64)
            if restart == 0:
                phases0 = 0.25 * np.sin(grid)
            else:
                phases0 = rng.uniform(-np.pi, np.pi, size=slots)
            variables0 = np.concatenate([phases0, np.array([0.0, 0.0], dtype=np.float64)])
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * slots + [(0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)],
                options={"maxiter": int(self.config.max_iter)},
            )
            candidate = self._result_from_variables(
                result.x,
                num_iter=int(result.nit),
                num_fid_func_calls=int(result.nfev),
                wall_time=time.perf_counter() - started_at,
                success=bool(result.success),
                termination_reason=str(result.message),
            )
            if self.config.show_progress:
                print(
                    f"[ma2023-six-level] restart {restart + 1}/{self.config.num_restarts} "
                    f"F={candidate.fidelity:.8f} Fpro={candidate.process_fidelity:.8f} "
                    f"L={candidate.leakage:.3e}",
                    flush=True,
                )
            if best is None or candidate.fidelity > best.fidelity:
                best = candidate
        assert best is not None
        return best

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        phases = wrap_phase(variables[:-2])
        theta0 = float(variables[-2])
        theta1 = float(variables[-1])
        ctrl_x, ctrl_y = controls_from_envelope_phase(self.envelope, phases)
        fidelity, fpro, leakage, grad_x, grad_y, theta0_grad, theta1_grad = self._fidelity_and_gradients(
            ctrl_x,
            ctrl_y,
            theta0,
            theta1,
        )
        phase_cost, phase_grad = phase_regularization(
            phases,
            self.config.phase_smoothness_weight,
            self.config.phase_curvature_weight,
        )
        fidelity_phase_gradient = (
            -grad_x * self.envelope * np.sin(phases)
            + grad_y * self.envelope * np.cos(phases)
        )
        objective = 1.0 - fidelity + phase_cost
        gradient = np.concatenate(
            [
                -fidelity_phase_gradient + phase_grad,
                np.array([-theta0_grad, -theta1_grad], dtype=np.float64),
            ]
        )
        return float(objective), gradient

    def evolve_basis(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, initial_index: int) -> np.ndarray:
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[int(initial_index)] = 1.0
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            state = expm(self.config.dt * self._generator(float(x_value), float(y_value))) @ state
        return state

    def _fidelity_and_gradients(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        theta0: float,
        theta1: float,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
        slots = int(ctrl_x.size)
        final_states: list[np.ndarray] = []
        prefixes_by_state: list[list[np.ndarray]] = []
        slice_propags: list[np.ndarray] = []

        for initial_index in self.initial_indices:
            state = np.zeros(self.dimension, dtype=np.complex128)
            state[int(initial_index)] = 1.0
            prefixes = [state]
            if not slice_propags:
                for x_value, y_value in zip(ctrl_x, ctrl_y):
                    slice_propags.append(expm(self.config.dt * self._generator(float(x_value), float(y_value))))
            for propagator in slice_propags:
                state = propagator @ state
                prefixes.append(state)
            final_states.append(state)
            prefixes_by_state.append(prefixes)

        z = np.array(
            [final_states[index][basis_index] for index, basis_index in enumerate(self.initial_indices)],
            dtype=np.complex128,
        )
        weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        signs = np.array([1.0, 1.0, -1.0], dtype=np.complex128)
        target_conj = np.array(
            [
                np.exp(-1j * theta0),
                np.exp(-1j * (theta0 + theta1)),
                np.exp(-1j * (theta0 + 2.0 * theta1)),
            ],
            dtype=np.complex128,
        )
        trace = np.sum(weights * signs * target_conj * z)
        process_fidelity = float(abs(trace) ** 2 / 16.0)
        population = float(np.sum(weights * np.abs(z) ** 2))
        leakage = float(1.0 - population / 4.0)
        fidelity = float((4.0 * process_fidelity + 1.0 - leakage) / 5.0)

        suffix_propags: list[np.ndarray] = [np.eye(self.dimension, dtype=np.complex128) for _ in range(slots)]
        suffix = np.eye(self.dimension, dtype=np.complex128)
        for index in range(slots - 1, -1, -1):
            suffix_propags[index] = suffix
            suffix = suffix @ slice_propags[index]

        grad_x = np.zeros(slots, dtype=np.float64)
        grad_y = np.zeros(slots, dtype=np.float64)
        for slot, (x_value, y_value) in enumerate(zip(ctrl_x, ctrl_y)):
            generator = self._generator(float(x_value), float(y_value))
            du_x = expm_frechet(self.config.dt * generator, self.config.dt * self.g_x, compute_expm=False)
            du_y = expm_frechet(self.config.dt * generator, self.config.dt * self.g_y, compute_expm=False)
            for state_index, basis_index in enumerate(self.initial_indices):
                prefix = prefixes_by_state[state_index][slot]
                dz_x = (suffix_propags[slot] @ du_x @ prefix)[basis_index]
                dz_y = (suffix_propags[slot] @ du_y @ prefix)[basis_index]
                grad_x[slot] += self._fidelity_z_gradient(
                    dz_x,
                    z,
                    trace,
                    target_conj,
                    signs,
                    weights,
                    state_index,
                )
                grad_y[slot] += self._fidelity_z_gradient(
                    dz_y,
                    z,
                    trace,
                    target_conj,
                    signs,
                    weights,
                    state_index,
                )

        dtrace_theta0 = -1j * trace
        dtrace_theta1 = (
            -2j * target_conj[1] * z[1]
            + 2j * target_conj[2] * z[2]
        )
        theta0_grad = float((4.0 / 5.0) * np.real(np.conj(trace) * dtrace_theta0) / 8.0)
        theta1_grad = float((4.0 / 5.0) * np.real(np.conj(trace) * dtrace_theta1) / 8.0)
        return fidelity, process_fidelity, leakage, grad_x, grad_y, theta0_grad, theta1_grad

    def _fidelity_z_gradient(
        self,
        dz: complex,
        z: np.ndarray,
        trace: complex,
        target_conj: np.ndarray,
        signs: np.ndarray,
        weights: np.ndarray,
        state_index: int,
    ) -> float:
        dtrace = weights[state_index] * signs[state_index] * target_conj[state_index] * dz
        d_fpro = np.real(np.conj(trace) * dtrace) / 8.0
        d_population = 2.0 * weights[state_index] * np.real(np.conj(z[state_index]) * dz)
        return float((4.0 * d_fpro + d_population / 4.0) / 5.0)

    def _result_from_variables(
        self,
        variables: np.ndarray,
        *,
        num_iter: int,
        num_fid_func_calls: int,
        wall_time: float,
        success: bool,
        termination_reason: str,
    ) -> Ma2023SixLevelGRAPEResult:
        phases = wrap_phase(variables[:-2])
        theta0 = float(variables[-2])
        theta1 = float(variables[-1])
        ctrl_x, ctrl_y = controls_from_envelope_phase(self.envelope, phases)
        fidelity, process_fidelity, leakage, *_ = self._fidelity_and_gradients(ctrl_x, ctrl_y, theta0, theta1)
        amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2)
        objective = float(self.objective_and_gradient(variables)[0])
        return Ma2023SixLevelGRAPEResult(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            amplitudes=amplitudes,
            phases=phases,
            theta0=theta0,
            theta1=theta1,
            fidelity=fidelity,
            process_fidelity=process_fidelity,
            leakage=leakage,
            objective=objective,
            num_iter=num_iter,
            num_fid_func_calls=num_fid_func_calls,
            wall_time=wall_time,
            success=success,
            termination_reason=termination_reason,
        )

    def _decay_matrix(self) -> np.ndarray:
        decay = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for operator in self.model.collapse_operators():
            matrix = np.asarray(operator.full(), dtype=np.complex128)
            decay += matrix.conj().T @ matrix
        return decay

    def _generator(self, ctrl_x: float, ctrl_y: float) -> np.ndarray:
        return self.g_d + ctrl_x * self.g_x + ctrl_y * self.g_y


class Ma2023SixLevelChebyshevPhaseRateOptimizer(Ma2023SixLevelPhaseOptimizer):
    """GRAPE optimizer using the Ma 2023 Chebyshev phase-rate parameterization.

    The optimized control variables are ``c_0..c_n, phi_0, theta_0, theta_1``.
    The phase rate is sampled as

        d phi / dt = sum_n c_n T_n(2 t / T - 1)

    and integrated to the piecewise-constant slot phases used by the propagator.
    """

    def __init__(
        self,
        *,
        model: Ma2023PerfectBlockadeSixLevelModel,
        config: Ma2023SixLevelGRAPEConfig,
        envelope: np.ndarray,
    ) -> None:
        super().__init__(model=model, config=config, envelope=envelope)
        self.chebyshev_degree = int(config.chebyshev_degree)
        self.rate_basis, self.phase_basis = self._build_chebyshev_phase_bases()

    def optimize(self, initial_variables: np.ndarray | None = None) -> Ma2023SixLevelGRAPEResult:
        rng = np.random.default_rng(self.config.seed)
        best: Ma2023SixLevelGRAPEResult | None = None
        started_at = time.perf_counter()
        coefficient_count = self.chebyshev_degree + 1
        coefficient_bound = self.config.chebyshev_coefficient_bound
        coefficient_bounds = (
            [(None, None)] * coefficient_count
            if coefficient_bound is None
            else [(-float(coefficient_bound), float(coefficient_bound))] * coefficient_count
        )

        for restart in range(self.config.num_restarts):
            if restart == 0 and initial_variables is not None:
                variables0 = np.asarray(initial_variables, dtype=np.float64)
            elif restart == 0:
                coefficients0 = np.zeros(coefficient_count, dtype=np.float64)
                if coefficient_count > 1:
                    coefficients0[1] = float(self.config.chebyshev_init_scale)
                variables0 = np.concatenate([coefficients0, np.array([0.0, 0.0, 0.0], dtype=np.float64)])
            else:
                scale = float(self.config.chebyshev_init_scale)
                coefficients0 = rng.normal(0.0, scale, size=coefficient_count)
                variables0 = np.concatenate([coefficients0, np.array([0.0, 0.0, 0.0], dtype=np.float64)])
            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                bounds=coefficient_bounds + [(-np.pi, np.pi), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)],
                options={"maxiter": int(self.config.max_iter)},
            )
            candidate = self._result_from_variables(
                result.x,
                num_iter=int(result.nit),
                num_fid_func_calls=int(result.nfev),
                wall_time=time.perf_counter() - started_at,
                success=bool(result.success),
                termination_reason=str(result.message),
            )
            if self.config.show_progress:
                print(
                    f"[ma2023-six-level-cheb] restart {restart + 1}/{self.config.num_restarts} "
                    f"F={candidate.fidelity:.8f} Fpro={candidate.process_fidelity:.8f} "
                    f"L={candidate.leakage:.3e}",
                    flush=True,
                )
            if best is None or candidate.fidelity > best.fidelity:
                best = candidate
        assert best is not None
        return best

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, phase_origin, theta0, theta1 = self._unpack_chebyshev_variables(variables)
        phases = self.phases_from_coefficients(coefficients, phase_origin)
        ctrl_x, ctrl_y = controls_from_envelope_phase(self.envelope, phases)
        fidelity, _fpro, _leakage, grad_x, grad_y, theta0_grad, theta1_grad = self._fidelity_and_gradients(
            ctrl_x,
            ctrl_y,
            theta0,
            theta1,
        )
        phase_cost, phase_grad = phase_regularization(
            wrap_phase(phases),
            self.config.phase_smoothness_weight,
            self.config.phase_curvature_weight,
        )
        fidelity_phase_gradient = (
            -grad_x * self.envelope * np.sin(phases)
            + grad_y * self.envelope * np.cos(phases)
        )
        objective_phase_gradient = -fidelity_phase_gradient + phase_grad
        coefficient_gradient = self.phase_basis.T @ objective_phase_gradient
        phase_origin_gradient = float(np.sum(objective_phase_gradient))
        gradient = np.concatenate(
            [
                coefficient_gradient,
                np.array(
                    [
                        phase_origin_gradient,
                        -theta0_grad,
                        -theta1_grad,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        objective = 1.0 - fidelity + phase_cost
        return float(objective), gradient

    def phases_from_coefficients(self, coefficients: np.ndarray, phase_origin: float) -> np.ndarray:
        return float(phase_origin) + self.phase_basis @ np.asarray(coefficients, dtype=np.float64)

    def phase_rates_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        return self.rate_basis @ np.asarray(coefficients, dtype=np.float64)

    def variables_from_slot_phases(
        self,
        phases: np.ndarray,
        *,
        theta0: float = 0.0,
        theta1: float = 0.0,
    ) -> np.ndarray:
        unwrapped_phases = np.unwrap(np.asarray(phases, dtype=np.float64))
        design = np.column_stack([self.phase_basis, np.ones(self.phase_basis.shape[0], dtype=np.float64)])
        solution, *_ = np.linalg.lstsq(design, unwrapped_phases, rcond=None)
        coefficients = solution[:-1]
        phase_origin = wrap_phase(np.array([solution[-1]], dtype=np.float64))[0]
        return np.concatenate(
            [
                coefficients,
                np.array([phase_origin, float(theta0), float(theta1)], dtype=np.float64),
            ]
        )

    def _result_from_variables(
        self,
        variables: np.ndarray,
        *,
        num_iter: int,
        num_fid_func_calls: int,
        wall_time: float,
        success: bool,
        termination_reason: str,
    ) -> Ma2023SixLevelGRAPEResult:
        coefficients, phase_origin, theta0, theta1 = self._unpack_chebyshev_variables(variables)
        phases = self.phases_from_coefficients(coefficients, phase_origin)
        ctrl_x, ctrl_y = controls_from_envelope_phase(self.envelope, phases)
        fidelity, process_fidelity, leakage, *_ = self._fidelity_and_gradients(ctrl_x, ctrl_y, theta0, theta1)
        amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2)
        objective = float(self.objective_and_gradient(variables)[0])
        return Ma2023SixLevelGRAPEResult(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            amplitudes=amplitudes,
            phases=wrap_phase(phases),
            theta0=theta0,
            theta1=theta1,
            fidelity=fidelity,
            process_fidelity=process_fidelity,
            leakage=leakage,
            objective=objective,
            num_iter=num_iter,
            num_fid_func_calls=num_fid_func_calls,
            wall_time=wall_time,
            success=success,
            termination_reason=termination_reason,
            parameterization="chebyshev_phase_rate",
            phase_rate_coefficients=coefficients,
            phase_origin=phase_origin,
        )

    def _build_chebyshev_phase_bases(self) -> tuple[np.ndarray, np.ndarray]:
        slots = int(self.envelope.size)
        centers = (np.arange(slots, dtype=np.float64) + 0.5) / float(slots)
        chebyshev_x = 2.0 * centers - 1.0
        rate_basis = np.polynomial.chebyshev.chebvander(chebyshev_x, self.chebyshev_degree)
        phase_basis = self.config.dt * (np.cumsum(rate_basis, axis=0) - 0.5 * rate_basis)
        return rate_basis.astype(np.float64), phase_basis.astype(np.float64)

    def _unpack_chebyshev_variables(self, variables: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        coefficient_count = self.chebyshev_degree + 1
        coefficients = np.asarray(variables[:coefficient_count], dtype=np.float64)
        phase_origin = float(variables[coefficient_count])
        theta0 = float(variables[coefficient_count + 1])
        theta1 = float(variables[coefficient_count + 2])
        return coefficients, phase_origin, theta0, theta1
