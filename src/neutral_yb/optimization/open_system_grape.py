from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import qutip
from qutip_qtrl import pulseoptim


@dataclass(frozen=True)
class OpenSystemGRAPEConfig:
    num_tslots: int = 40
    evo_time: float = 8.8
    max_iter: int = 120
    max_wall_time: float = 600.0
    fid_err_targ: float = 1e-3
    min_grad: float = 1e-8
    num_restarts: int = 3
    seed: int = 17
    init_pulse_type: str = "SINE"
    pulse_scaling: float = 0.3
    pulse_offset: float = 0.0
    target_theta: float = 0.0
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
    num_iter: int
    num_fid_func_calls: int
    wall_time: float
    termination_reason: str
    evo_time: float
    num_tslots: int

    def to_json(self) -> dict[str, float | int | str | list[float]]:
        return {
            "ctrl_x": [float(x) for x in self.ctrl_x],
            "ctrl_y": [float(x) for x in self.ctrl_y],
            "amplitudes": [float(x) for x in self.amplitudes],
            "phases": [float(x) for x in self.phases],
            "target_theta": float(self.target_theta),
            "optimized_theta": float(self.optimized_theta),
            "fid_err": float(self.fid_err),
            "probe_fidelity": float(self.probe_fidelity),
            "num_iter": int(self.num_iter),
            "num_fid_func_calls": int(self.num_fid_func_calls),
            "wall_time": float(self.wall_time),
            "termination_reason": self.termination_reason,
            "evo_time": float(self.evo_time),
            "num_tslots": int(self.num_tslots),
        }


class OpenSystemGRAPEOptimizer:
    """Liouvillian GRAPE wrapper for the v4 open-system model."""

    def __init__(self, model, config: OpenSystemGRAPEConfig):
        self.model = model
        self.config = config

    def optimize(self) -> OpenSystemGRAPEResult:
        best: OpenSystemGRAPEResult | None = None
        amp_bound = self.model.control_amplitude_bound()

        for restart in range(self.config.num_restarts):
            if self.config.show_progress:
                print(
                    f"[open-grape] restart {restart + 1}/{self.config.num_restarts} "
                    f"T={self.config.evo_time:.3f} slots={self.config.num_tslots}",
                    flush=True,
                )
            np.random.seed(self.config.seed + restart)
            started_at = time.perf_counter()
            optim_result = pulseoptim.optimize_pulse(
                drift=self.model.drift_liouvillian(),
                ctrls=list(self.model.control_liouvillians()),
                initial=self.model.initial_superoperator(),
                target=self.model.target_superoperator(self.config.target_theta),
                num_tslots=self.config.num_tslots,
                evo_time=self.config.evo_time,
                amp_lbound=-amp_bound,
                amp_ubound=amp_bound,
                fid_err_targ=self.config.fid_err_targ,
                min_grad=self.config.min_grad,
                max_iter=self.config.max_iter,
                max_wall_time=self.config.max_wall_time,
                alg="GRAPE",
                optim_method="FMIN_L_BFGS_B",
                dyn_type="GEN_MAT",
                prop_type="FRECHET",
                fid_type="TRACEDIFF",
                init_pulse_type=self.config.init_pulse_type,
                pulse_scaling=self.config.pulse_scaling,
                pulse_offset=self.config.pulse_offset,
                log_level=0,
                gen_stats=False,
            )
            wall_time = time.perf_counter() - started_at
            ctrls = np.asarray(optim_result.final_amps, dtype=np.float64)
            candidate = self._result_from_controls(
                ctrls[:, 0],
                ctrls[:, 1],
                optim_result.fid_err,
                int(optim_result.num_iter),
                int(optim_result.num_fid_func_calls),
                wall_time,
                str(optim_result.termination_reason),
            )
            if self.config.show_progress:
                print(
                    f"[open-grape] restart {restart + 1}/{self.config.num_restarts} "
                    f"fid_err={candidate.fid_err:.6e} probe_F={candidate.probe_fidelity:.6f} "
                    f"time={candidate.wall_time:7.1f}s",
                    flush=True,
                )
            if best is None or self._is_better(candidate, best):
                best = candidate
        assert best is not None
        return best

    def evolve_probe_states(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> list[qutip.Qobj]:
        dt = self.config.dt
        tlist = np.linspace(0.0, self.config.evo_time, self.config.num_tslots + 1)
        h_terms = self.model.evolution_hamiltonian_terms(ctrl_x, ctrl_y, dt)
        c_ops = self.model.collapse_operators()
        final_states: list[qutip.Qobj] = []
        for probe_ket, _ in self.model.probe_kets(theta=0.0):
            rho0 = qutip.ket2dm(probe_ket)
            result = qutip.mesolve(
                h_terms,
                rho0,
                tlist,
                c_ops=c_ops,
                e_ops=None,
                options={"store_states": False, "store_final_state": True},
            )
            final_states.append(result.final_state)
        return final_states

    def benchmark_probe_evolution(self, ctrl_x: np.ndarray, ctrl_y: np.ndarray, repeats: int = 3) -> float:
        self.evolve_probe_states(ctrl_x, ctrl_y)
        started_at = time.perf_counter()
        for _ in range(repeats):
            self.evolve_probe_states(ctrl_x, ctrl_y)
        return (time.perf_counter() - started_at) / repeats

    def save_result(self, result: OpenSystemGRAPEResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    def _result_from_controls(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        fid_err: float,
        num_iter: int,
        num_fid_func_calls: int,
        wall_time: float,
        termination_reason: str,
    ) -> OpenSystemGRAPEResult:
        amplitudes, phases = self.model.control_cartesian_to_polar(ctrl_x, ctrl_y)
        final_states = self.evolve_probe_states(ctrl_x, ctrl_y)
        theta, probe_fidelity = self.model.optimize_theta_for_probe_states(final_states)
        return OpenSystemGRAPEResult(
            ctrl_x=np.asarray(ctrl_x, dtype=np.float64),
            ctrl_y=np.asarray(ctrl_y, dtype=np.float64),
            amplitudes=amplitudes,
            phases=phases,
            target_theta=float(self.config.target_theta),
            optimized_theta=float(theta),
            fid_err=float(fid_err),
            probe_fidelity=float(probe_fidelity),
            num_iter=num_iter,
            num_fid_func_calls=num_fid_func_calls,
            wall_time=float(wall_time),
            termination_reason=termination_reason,
            evo_time=float(self.config.evo_time),
            num_tslots=int(self.config.num_tslots),
        )

    @staticmethod
    def _is_better(left: OpenSystemGRAPEResult, right: OpenSystemGRAPEResult) -> bool:
        if left.probe_fidelity > right.probe_fidelity + 1e-8:
            return True
        if abs(left.probe_fidelity - right.probe_fidelity) <= 1e-8 and left.fid_err < right.fid_err:
            return True
        return False
