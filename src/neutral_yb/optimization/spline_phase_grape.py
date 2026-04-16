from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    GlobalPhaseOptimizationResult,
    PaperGlobalPhaseOptimizer,
    TimeOptimalScanResult,
)


@dataclass(frozen=True)
class SplinePhaseOptimizationConfig:
    num_tslots: int = 120
    num_nodes: int = 16
    evo_time: float = 8.5
    max_iter: int = 300
    phase_seed: int = 29
    init_phase_spread: float = 0.35
    fidelity_target: float = 0.9999
    smoothness_weight: float = 0.01
    curvature_weight: float = 0.02
    node_curvature_weight: float = 0.01
    num_restarts: int = 4
    show_progress: bool = False


@dataclass(frozen=True)
class SplinePhaseOptimizationResult:
    node_phases: np.ndarray
    slice_phases: np.ndarray
    theta: float
    fidelity: float
    objective: float
    iterations: int
    success: bool
    message: str
    evo_time: float
    num_tslots: int
    num_nodes: int
    smoothness_cost: float
    curvature_cost: float
    node_curvature_cost: float

    def to_json(self) -> dict[str, float | int | bool | str | list[float]]:
        return {
            "node_phases": [float(x) for x in self.node_phases],
            "phases": [float(x) for x in self.slice_phases],
            "theta": float(self.theta),
            "fidelity": float(self.fidelity),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
            "num_nodes": int(self.num_nodes),
            "smoothness_cost": float(self.smoothness_cost),
            "curvature_cost": float(self.curvature_cost),
            "node_curvature_cost": float(self.node_curvature_cost),
        }


class SplinePhaseOptimizer:
    """Optimize a single phase profile through cubic-spline control nodes."""

    def __init__(self, model, config: SplinePhaseOptimizationConfig):
        self.model = model
        self.config = config
        self.base_optimizer = PaperGlobalPhaseOptimizer(
            model,
            GlobalPhaseOptimizationConfig(
                num_tslots=config.num_tslots,
                evo_time=config.evo_time,
                max_iter=config.max_iter,
                phase_seed=config.phase_seed,
                init_phase_spread=config.init_phase_spread,
                fidelity_target=config.fidelity_target,
                smoothness_weight=config.smoothness_weight,
                curvature_weight=config.curvature_weight,
                num_restarts=1,
                show_progress=False,
            ),
        )
        self.node_times = np.linspace(0.0, config.evo_time, config.num_nodes)
        self.slice_times = np.linspace(0.0, config.evo_time, config.num_tslots)
        self.interpolation_matrix = self._build_interpolation_matrix()

    def initial_nodes(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed)
        nodes = rng.normal(0.0, self.config.init_phase_spread, size=self.config.num_nodes)
        return np.mod(nodes, 2.0 * np.pi)

    def optimize(
        self,
        initial_nodes: np.ndarray | None = None,
        initial_theta: float = 0.0,
    ) -> SplinePhaseOptimizationResult:
        best_result: SplinePhaseOptimizationResult | None = None
        base_nodes = self.initial_nodes() if initial_nodes is None else np.asarray(initial_nodes, dtype=np.float64)

        for restart in range(self.config.num_restarts):
            restart_started_at = time.perf_counter()
            nodes0 = base_nodes if restart == 0 else self._jittered_initial_nodes(base_nodes, restart)
            variables0 = np.concatenate([np.asarray(nodes0, dtype=np.float64), np.array([initial_theta])])
            callback_state = {"iterations": 0}

            def callback(_: np.ndarray) -> None:
                callback_state["iterations"] += 1

            result = minimize(
                fun=self.objective_and_gradient,
                x0=variables0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.config.max_iter},
                callback=callback,
            )

            node_phases = result.x[:-1]
            theta = float(np.mod(result.x[-1], 2.0 * np.pi))
            slice_phases = self.slice_phases_from_nodes(node_phases)
            final_state = self.base_optimizer.final_state(slice_phases)
            fidelity = self.model.phase_gate_fidelity(final_state, theta)
            smoothness_cost = self.base_optimizer._smoothness_cost(slice_phases)
            curvature_cost = self.base_optimizer._curvature_cost(slice_phases)
            node_curvature_cost = self._node_curvature_cost(node_phases)

            candidate = SplinePhaseOptimizationResult(
                node_phases=node_phases,
                slice_phases=np.mod(slice_phases, 2.0 * np.pi),
                theta=theta,
                fidelity=float(fidelity),
                objective=float(
                    1.0
                    - fidelity
                    + self.config.node_curvature_weight * node_curvature_cost
                ),
                iterations=int(callback_state["iterations"]),
                success=bool(result.success),
                message=str(result.message),
                evo_time=float(self.config.evo_time),
                num_tslots=int(self.config.num_tslots),
                num_nodes=int(self.config.num_nodes),
                smoothness_cost=float(smoothness_cost),
                curvature_cost=float(curvature_cost),
                node_curvature_cost=float(node_curvature_cost),
            )
            if self.config.show_progress:
                elapsed = time.perf_counter() - restart_started_at
                print(
                    f"[spline-opt] restart {restart + 1}/{self.config.num_restarts} "
                    f"iter={callback_state['iterations']:4d} elapsed={elapsed:7.1f}s "
                    f"F={candidate.fidelity:.9f} smooth={candidate.smoothness_cost:.6f} "
                    f"curv={candidate.curvature_cost:.6f} node_curv={candidate.node_curvature_cost:.6f}",
                    flush=True,
                )
            if best_result is None or self._is_better(candidate, best_result):
                best_result = candidate

        assert best_result is not None
        return best_result

    def scan_durations(
        self,
        durations: list[float],
        initial_nodes: np.ndarray | None = None,
    ) -> tuple[TimeOptimalScanResult, list[SplinePhaseOptimizationResult]]:
        nodes = self.initial_nodes() if initial_nodes is None else np.asarray(initial_nodes, dtype=np.float64)
        theta = 0.0
        results: list[SplinePhaseOptimizationResult] = []
        fidelities: list[float] = []
        scan_started_at = time.perf_counter()

        for duration in durations:
            started_at = time.perf_counter()
            optimizer = SplinePhaseOptimizer(
                self.model,
                SplinePhaseOptimizationConfig(
                    num_tslots=self.config.num_tslots,
                    num_nodes=self.config.num_nodes,
                    evo_time=duration,
                    max_iter=self.config.max_iter,
                    phase_seed=self.config.phase_seed,
                    init_phase_spread=self.config.init_phase_spread,
                    fidelity_target=self.config.fidelity_target,
                    smoothness_weight=self.config.smoothness_weight,
                    curvature_weight=self.config.curvature_weight,
                    node_curvature_weight=self.config.node_curvature_weight,
                    num_restarts=self.config.num_restarts,
                    show_progress=self.config.show_progress,
                ),
            )
            result = optimizer.optimize(nodes, theta)
            results.append(result)
            fidelities.append(result.fidelity)
            nodes = result.node_phases
            theta = result.theta
            if self.config.show_progress:
                index = len(results)
                step_elapsed = time.perf_counter() - started_at
                total_elapsed = time.perf_counter() - scan_started_at
                avg_per_step = total_elapsed / index
                remaining = avg_per_step * (len(durations) - index)
                print(
                    f"[spline-scan] {index}/{len(durations)} "
                    f"T={duration:.3f} F={result.fidelity:.9f} "
                    f"step={step_elapsed:7.1f}s elapsed={total_elapsed:7.1f}s "
                    f"remaining_T={len(durations)-index:2d} eta={remaining:7.1f}s",
                    flush=True,
                )

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

    def save_result(self, result: SplinePhaseOptimizationResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def save_scan(self, result: TimeOptimalScanResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def trajectory(self, node_phases: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.base_optimizer.trajectory(self.slice_phases_from_nodes(node_phases))

    def objective_and_gradient(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        node_phases = np.asarray(variables[:-1], dtype=np.float64)
        theta = float(variables[-1])
        slice_phases = self.slice_phases_from_nodes(node_phases)
        base_variables = np.concatenate([slice_phases, np.array([theta])])
        objective, base_gradient = self.base_optimizer.objective_and_gradient(base_variables)
        node_curvature_cost = self._node_curvature_cost(node_phases)
        node_gradient = self.interpolation_matrix.T @ base_gradient[:-1]
        if self.config.node_curvature_weight > 0.0:
            node_gradient += self.config.node_curvature_weight * self._node_curvature_gradient(node_phases)
        total_objective = float(objective + self.config.node_curvature_weight * node_curvature_cost)
        gradient = np.concatenate([node_gradient, np.array([base_gradient[-1]])])
        return total_objective, gradient

    def slice_phases_from_nodes(self, node_phases: np.ndarray) -> np.ndarray:
        return self.interpolation_matrix @ np.asarray(node_phases, dtype=np.float64)

    def _build_interpolation_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.config.num_tslots, self.config.num_nodes), dtype=np.float64)
        for node_index in range(self.config.num_nodes):
            basis = np.zeros(self.config.num_nodes, dtype=np.float64)
            basis[node_index] = 1.0
            spline = CubicSpline(self.node_times, basis, bc_type="natural")
            matrix[:, node_index] = spline(self.slice_times)
        return matrix

    def _jittered_initial_nodes(self, base_nodes: np.ndarray, restart: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.phase_seed + 1000 * restart)
        jitter = rng.normal(0.0, self.config.init_phase_spread * 0.35, size=base_nodes.shape[0])
        return base_nodes + jitter

    @staticmethod
    def _node_curvature_cost(node_phases: np.ndarray) -> float:
        if len(node_phases) < 3:
            return 0.0
        curvature = node_phases[2:] - 2.0 * node_phases[1:-1] + node_phases[:-2]
        return float(np.mean(curvature**2))

    @staticmethod
    def _node_curvature_gradient(node_phases: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(node_phases)
        if len(node_phases) < 3:
            return grad
        curvature = node_phases[2:] - 2.0 * node_phases[1:-1] + node_phases[:-2]
        scale = 2.0 / curvature.size
        grad[:-2] += curvature * scale
        grad[1:-1] += -2.0 * curvature * scale
        grad[2:] += curvature * scale
        return grad

    @staticmethod
    def _is_better(left: SplinePhaseOptimizationResult, right: SplinePhaseOptimizationResult) -> bool:
        if left.fidelity > right.fidelity + 1e-8:
            return True
        if abs(left.fidelity - right.fidelity) <= 1e-8 and left.objective < right.objective:
            return True
        return False
