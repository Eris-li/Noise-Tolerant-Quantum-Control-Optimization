from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import (
    build_yb171_v3_calibrated_model,
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    yb171_gate_time_ns_to_dimensionless,
    yb171_v4_default_omega_max_hz,
)
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.amplitude_phase_grape import (
    AmplitudePhaseOptimizationConfig,
    AmplitudePhaseOptimizer,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def build_closed_optimizer() -> AmplitudePhaseOptimizer:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    gate_time_ns = 136.0
    model = build_yb171_v3_calibrated_model(effective_rabi_hz=omega_max_hz)
    return AmplitudePhaseOptimizer(
        model=model,
        config=AmplitudePhaseOptimizationConfig(
            num_tslots=40,
            evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=omega_max_hz),
            max_iter=3,
            seed=17,
            init_phase_spread=0.35,
            init_amplitude_scale=0.75,
            num_restarts=1,
        ),
    )


def build_open_optimizer() -> OpenSystemGRAPEOptimizer:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    gate_time_ns = 136.0
    model = build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz)
    return OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=omega_max_hz),
            max_iter=2,
            num_restarts=1,
            seed=17,
            init_control_scale=0.08,
        ),
        ensemble_models=build_yb171_v4_quasistatic_ensemble(
            ensemble_size=3,
            seed=17,
            effective_rabi_hz=omega_max_hz,
        ),
    )


def average_time(callable_obj, repeats: int) -> float:
    callable_obj()
    started_at = time.perf_counter()
    for _ in range(repeats):
        callable_obj()
    return (time.perf_counter() - started_at) / repeats


def main() -> None:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    gate_time_ns = 136.0
    closed = build_closed_optimizer()
    amplitudes, phases = closed.initial_guess()
    variables = np.concatenate([amplitudes, phases, np.array([0.3])])

    closed_final_state_avg = average_time(lambda: closed.final_state(amplitudes, phases), repeats=10)
    closed_obj_grad_avg = average_time(lambda: closed.objective_and_gradient(variables), repeats=5)
    closed_started_at = time.perf_counter()
    closed_result = closed.optimize(initial_amplitudes=amplitudes, initial_phases=phases, initial_theta=0.3)
    closed_opt_wall = time.perf_counter() - closed_started_at

    open_system = build_open_optimizer()
    open_bound = open_system.model.control_amplitude_bound()
    ctrl_x = np.full(open_system.config.num_tslots, 0.35 * open_bound, dtype=np.float64)
    ctrl_y = np.zeros(open_system.config.num_tslots, dtype=np.float64)
    open_probe_avg = open_system.benchmark_probe_evolution(ctrl_x, ctrl_y, repeats=3)
    open_started_at = time.perf_counter()
    open_result = open_system.optimize()
    open_opt_wall = time.perf_counter() - open_started_at

    summary = {
        "gate_time_ns": gate_time_ns,
        "omega_max_hz": omega_max_hz,
        "omega_max_mhz": omega_max_hz / 1e6,
        "closed_system": {
            "num_tslots": closed.config.num_tslots,
            "dimensionless_gate_time": closed.config.evo_time,
            "final_state_avg_s": closed_final_state_avg,
            "objective_and_gradient_avg_s": closed_obj_grad_avg,
            "optimize_wall_s": closed_opt_wall,
            "optimize_fidelity": closed_result.fidelity,
        },
        "open_system": {
            "num_tslots": open_system.config.num_tslots,
            "dimensionless_gate_time": open_system.config.evo_time,
            "probe_evolution_avg_s": open_probe_avg,
            "optimize_wall_s": open_opt_wall,
            "optimize_probe_fidelity": open_result.probe_fidelity,
            "optimize_channel_fidelity": open_result.probe_fidelity,
            "optimize_fid_err": open_result.fid_err,
        },
        "ratios": {
            "probe_over_closed_final": open_probe_avg / closed_final_state_avg,
            "probe_over_closed_objective_and_gradient": open_probe_avg / closed_obj_grad_avg,
            "optimize_over_closed_optimize": open_opt_wall / closed_opt_wall,
        },
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "benchmark_v4_open_system_vs_v3_closed.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Benchmark written to {destination}")


if __name__ == "__main__":
    main()
