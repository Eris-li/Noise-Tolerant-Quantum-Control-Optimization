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

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.amplitude_phase_grape import (
    AmplitudePhaseOptimizationConfig,
    AmplitudePhaseOptimizer,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def build_closed_optimizer() -> AmplitudePhaseOptimizer:
    model = TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
    )
    return AmplitudePhaseOptimizer(
        model=model,
        config=AmplitudePhaseOptimizationConfig(
            num_tslots=40,
            evo_time=8.5,
            max_iter=3,
            seed=17,
            init_phase_spread=0.35,
            init_amplitude_scale=0.75,
            num_restarts=1,
        ),
    )


def build_open_optimizer() -> OpenSystemGRAPEOptimizer:
    model = TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
        noise=TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.01,
            common_two_photon_detuning=0.004,
            differential_two_photon_detuning=0.003,
            doppler_detuning_01=0.002,
            doppler_detuning_11=0.004,
            lower_amplitude_scale=0.99,
            upper_amplitude_scale=0.99,
            intermediate_decay_rate=0.025,
            rydberg_decay_rate=0.015,
            intermediate_dephasing_rate=0.004,
            rydberg_dephasing_rate=0.01,
            extra_rydberg_leakage_rate=0.003,
            intermediate_branch_to_qubit=0.45,
            rydberg_branch_to_qubit=0.05,
        ),
    )
    return OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=40,
            evo_time=8.5,
            max_iter=3,
            max_wall_time=180.0,
            num_restarts=1,
            seed=17,
        ),
    )


def average_time(callable_obj, repeats: int) -> float:
    callable_obj()
    started_at = time.perf_counter()
    for _ in range(repeats):
        callable_obj()
    return (time.perf_counter() - started_at) / repeats


def main() -> None:
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
        "closed_system": {
            "num_tslots": closed.config.num_tslots,
            "evo_time": closed.config.evo_time,
            "final_state_avg_s": closed_final_state_avg,
            "objective_and_gradient_avg_s": closed_obj_grad_avg,
            "optimize_wall_s": closed_opt_wall,
            "optimize_fidelity": closed_result.fidelity,
        },
        "open_system": {
            "num_tslots": open_system.config.num_tslots,
            "evo_time": open_system.config.evo_time,
            "probe_evolution_avg_s": open_probe_avg,
            "optimize_wall_s": open_opt_wall,
            "optimize_probe_fidelity": open_result.probe_fidelity,
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
