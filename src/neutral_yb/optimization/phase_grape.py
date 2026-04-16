from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from neutral_yb.models.ideal_cz import IdealCZModel


@dataclass(frozen=True)
class PhaseOnlyGrapeConfig:
    num_tslots: int = 40
    evo_time: float = 4.0
    max_iter: int = 200
    leakage_weight: float = 5.0
    single_population_weight: float = 1.0
    double_population_weight: float = 1.0
    phase_error_weight: float = 2.0
    seed: int = 7
    init_phase_spread: float = 0.2

    @property
    def dt(self) -> float:
        return self.evo_time / self.num_tslots


@dataclass(frozen=True)
class OptimizationSummary:
    phases: np.ndarray
    gamma: float
    fidelity_to_cz: float
    leakage: float
    objective: float
    iterations: int
    success: bool
    message: str
    best_entangling_phase: float
    fidelity_to_entangling_family: float

    def to_json(self) -> dict[str, float | int | bool | str | list[float]]:
        return {
            "phases": [float(x) for x in self.phases],
            "gamma": float(self.gamma),
            "fidelity_to_cz": float(self.fidelity_to_cz),
            "leakage": float(self.leakage),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "best_entangling_phase": float(self.best_entangling_phase),
            "fidelity_to_entangling_family": float(self.fidelity_to_entangling_family),
        }


class PhaseOnlyGrapeOptimizer:
    def __init__(self, model: IdealCZModel, config: PhaseOnlyGrapeConfig):
        self.model = model
        self.config = config

        self.h_d = model.drift_hamiltonian().full()
        self.h_x, self.h_y = [operator.full() for operator in model.control_hamiltonians()]
        self.projector = model.computational_projector()
        self.target = model.computational_target()
        self.dimension = self.target.shape[0]
        self.omega_max = model.species.omega_max

    def initial_phases(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        baseline = np.zeros(self.config.num_tslots, dtype=np.float64)
        noise = rng.normal(scale=self.config.init_phase_spread, size=self.config.num_tslots)
        return np.mod(baseline + noise, 2.0 * np.pi)

    def optimize(self, initial_phases: np.ndarray | None = None) -> OptimizationSummary:
        if initial_phases is None:
            initial_phases = self.initial_phases()

        result = minimize(
            fun=lambda x: self.objective_and_gradient(x),
            x0=np.asarray(initial_phases, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 2.0 * np.pi)] * self.config.num_tslots,
            options={"maxiter": self.config.max_iter},
        )

        phases = np.mod(result.x, 2.0 * np.pi)
        metrics = self.metrics(phases)

        return OptimizationSummary(
            phases=phases,
            gamma=metrics["best_entangling_phase"],
            fidelity_to_cz=metrics["fidelity_to_cz"],
            leakage=metrics["leakage"],
            objective=metrics["objective"],
            iterations=result.nit,
            success=bool(result.success),
            message=str(result.message),
            best_entangling_phase=metrics["best_entangling_phase"],
            fidelity_to_entangling_family=metrics["fidelity_to_entangling_family"],
        )

    def save_summary(self, summary: OptimizationSummary, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary.to_json(), indent=2), encoding="utf-8")

    def trajectory(self, phases: np.ndarray, initial_state: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        phases = np.asarray(phases, dtype=np.float64)
        state = np.asarray(initial_state, dtype=np.complex128)
        states = [state]
        dt = self.config.dt
        times = np.linspace(0.0, self.config.evo_time, len(phases) + 1)

        for phase in phases:
            h_k = self._hamiltonian_from_phase(phase)
            u_k = expm(-1j * dt * h_k)
            state = u_k @ state
            states.append(state)

        return times, states

    def metrics(self, phases: np.ndarray) -> dict[str, float]:
        total_unitary, _ = self._propagate(phases)
        reduced = self.projector.conj().T @ total_unitary @ self.projector
        fidelity_to_cz = abs(np.trace(self.target.conj().T @ reduced)) ** 2 / (self.dimension**2)
        leakage = 1.0 - np.trace(reduced.conj().T @ reduced).real / self.dimension
        phase_objective, best_gamma, family_fidelity = self._phase_population_objective(reduced)
        objective = phase_objective + self.config.leakage_weight * leakage
        return {
            "fidelity_to_cz": float(fidelity_to_cz),
            "leakage": float(leakage),
            "objective": float(objective),
            "best_entangling_phase": float(best_gamma),
            "fidelity_to_entangling_family": float(family_fidelity),
        }

    def objective_and_gradient(self, phases: np.ndarray) -> tuple[float, np.ndarray]:
        phases = np.asarray(phases, dtype=np.float64)
        total_unitary, slice_unitaries = self._propagate(phases)
        forwards, backwards = self._forward_backward(slice_unitaries)
        reduced = self.projector.conj().T @ total_unitary @ self.projector

        leakage = 1.0 - np.trace(reduced.conj().T @ reduced).real / self.dimension
        phase_population_objective, _, _ = self._phase_population_objective(reduced)
        objective = phase_population_objective + self.config.leakage_weight * leakage
        gradient = np.zeros_like(phases)
        dt = self.config.dt

        z01 = reduced[1, 1]
        z10 = reduced[2, 2]
        z11 = reduced[3, 3]
        phase_error = np.angle(np.exp(1j * (np.angle(z11) - np.angle(z01) - np.angle(z10) - np.pi)))

        for index, phase in enumerate(phases):
            h_k = self._hamiltonian_from_phase(phase)
            dh_dphi = self._phase_derivative_hamiltonian(phase)

            du_k = expm_frechet(
                -1j * dt * h_k,
                -1j * dt * dh_dphi,
                compute_expm=False,
            )
            before = np.eye(h_k.shape[0], dtype=np.complex128) if index == 0 else forwards[index - 1]
            after = np.eye(h_k.shape[0], dtype=np.complex128) if index == len(phases) - 1 else backwards[index + 1]
            d_total = after @ du_k @ before
            d_reduced = self.projector.conj().T @ d_total @ self.projector

            d_leakage = -2.0 * np.real(np.trace(reduced.conj().T @ d_reduced)) / self.dimension
            dz01 = d_reduced[1, 1]
            dz10 = d_reduced[2, 2]
            dz11 = d_reduced[3, 3]

            d_single_population = -0.5 * (
                2.0 * np.real(np.conj(z01) * dz01) + 2.0 * np.real(np.conj(z10) * dz10)
            )
            d_double_population = -2.0 * np.real(np.conj(z11) * dz11)
            d_phase_error = np.imag(dz11 / z11 - dz01 / z01 - dz10 / z10)
            d_phase_objective = 2.0 * self.config.phase_error_weight * phase_error * d_phase_error

            gradient[index] = (
                self.config.single_population_weight * d_single_population
                + self.config.double_population_weight * d_double_population
                + d_phase_objective
                + self.config.leakage_weight * d_leakage
            )

        return float(objective), gradient

    def _propagate(self, phases: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        unitary = np.eye(self.h_d.shape[0], dtype=np.complex128)
        slice_unitaries: list[np.ndarray] = []
        dt = self.config.dt

        for phase in phases:
            h_k = self._hamiltonian_from_phase(phase)
            u_k = expm(-1j * dt * h_k)
            slice_unitaries.append(u_k)
            unitary = u_k @ unitary

        return unitary, slice_unitaries

    def _forward_backward(self, slice_unitaries: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        forwards: list[np.ndarray] = []
        current = np.eye(self.h_d.shape[0], dtype=np.complex128)
        for u_k in slice_unitaries:
            current = u_k @ current
            forwards.append(current)

        backwards: list[np.ndarray] = [np.eye(self.h_d.shape[0], dtype=np.complex128)] * len(slice_unitaries)
        current = np.eye(self.h_d.shape[0], dtype=np.complex128)
        for index in range(len(slice_unitaries) - 1, -1, -1):
            backwards[index] = current
            current = current @ slice_unitaries[index]

        return forwards, backwards

    def _hamiltonian_from_phase(self, phase: float) -> np.ndarray:
        return self.h_d + self.omega_max * (
            np.cos(phase) * self.h_x + np.sin(phase) * self.h_y
        )

    def _phase_derivative_hamiltonian(self, phase: float) -> np.ndarray:
        return self.omega_max * (-np.sin(phase) * self.h_x + np.cos(phase) * self.h_y)

    def _phase_population_objective(self, reduced: np.ndarray) -> tuple[float, float, float]:
        z01 = reduced[1, 1]
        z10 = reduced[2, 2]
        z11 = reduced[3, 3]

        single_population_term = self.config.single_population_weight * (
            2.0 - abs(z01) ** 2 - abs(z10) ** 2
        )
        double_population_term = self.config.double_population_weight * (1.0 - abs(z11) ** 2)

        gamma = 0.5 * (np.angle(z01) + np.angle(z10))
        phase_error = np.angle(np.exp(1j * (np.angle(z11) - np.angle(z01) - np.angle(z10) - np.pi)))
        phase_term = self.config.phase_error_weight * (phase_error**2)

        family_target = np.diag(
            [1.0, np.exp(1j * gamma), np.exp(1j * gamma), -np.exp(2j * gamma)]
        ).astype(np.complex128)
        family_fidelity = abs(np.trace(family_target.conj().T @ reduced)) ** 2 / (self.dimension**2)

        return float(single_population_term + double_population_term + phase_term), float(gamma), float(family_fidelity)
