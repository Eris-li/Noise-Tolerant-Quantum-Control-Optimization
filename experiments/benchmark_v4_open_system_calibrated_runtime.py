from __future__ import annotations

from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    yb171_gate_time_ns_to_dimensionless,
    yb171_v4_default_omega_max_hz,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def run_case(
    *,
    gate_time_ns: float,
    num_tslots: int,
    max_iter: int,
    num_restarts: int,
) -> dict[str, float | int]:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    ensemble_size = 3
    seed = 17
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=num_tslots,
            evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=omega_max_hz),
            max_iter=max_iter,
            num_restarts=num_restarts,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=0.999,
            show_progress=True,
        ),
        ensemble_models=build_yb171_v4_quasistatic_ensemble(
            ensemble_size=ensemble_size,
            seed=seed,
            effective_rabi_hz=omega_max_hz,
        ),
    )
    started_at = time.perf_counter()
    result = optimizer.optimize()
    wall_total = time.perf_counter() - started_at
    return {
        "gate_time_ns": gate_time_ns,
        "omega_max_hz": omega_max_hz,
        "omega_max_mhz": omega_max_hz / 1e6,
        "dimensionless_gate_time": optimizer.config.evo_time,
        "num_tslots": num_tslots,
        "max_iter": max_iter,
        "num_restarts": num_restarts,
        "ensemble_size": ensemble_size,
        "ensemble_seed": seed,
        "probe_fidelity": result.probe_fidelity,
        "channel_fidelity": result.probe_fidelity,
        "optimizer_reported_wall_time": result.wall_time,
        "total_wall_time": wall_total,
        "num_iter": result.num_iter,
        "num_fid_func_calls": result.num_fid_func_calls,
    }


def main() -> None:
    cases = [
        run_case(gate_time_ns=136.0, num_tslots=32, max_iter=1, num_restarts=1),
        run_case(gate_time_ns=136.0, num_tslots=100, max_iter=1, num_restarts=1),
    ]

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "benchmark_v4_open_system_calibrated_runtime.json"
    destination.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    print(json.dumps(cases, indent=2))
    print(f"Saved benchmark to {destination}")


if __name__ == "__main__":
    main()
