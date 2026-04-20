from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    summarize_yb171_v4_result,
    yb171_dimensionless_time_to_gate_time_ns,
    yb171_gate_time_ns_to_dimensionless,
    yb171_v4_default_omega_max_hz,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def main() -> None:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    ensemble_size = 3
    seed = 17
    gate_time_ns = yb171_dimensionless_time_to_gate_time_ns(8.5, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=omega_max_hz),
            max_iter=8,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            show_progress=True,
        ),
        ensemble_models=build_yb171_v4_quasistatic_ensemble(
            ensemble_size=ensemble_size,
            seed=seed,
            effective_rabi_hz=omega_max_hz,
        ),
    )
    result = optimizer.optimize()

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "two_photon_cz_v4_open_system_smoke.json"
    payload = summarize_yb171_v4_result(
        result=result,
        gate_time_ns=gate_time_ns,
        omega_max_hz=omega_max_hz,
        model=optimizer.model,
    )
    payload["ensemble_size"] = ensemble_size
    payload["ensemble_seed"] = seed
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Two-photon CZ v4 open-system smoke run completed")
    print(f"Gate time = {gate_time_ns:.3f} ns")
    print(f"Omega_max = {omega_max_hz / 1e6:.3f} MHz")
    print(f"Ensemble size = {ensemble_size}")
    print(f"Phase-gate fidelity = {result.probe_fidelity}")
    print(f"Objective = {result.objective}")
    print(f"Fid err = {result.fid_err}")
    print(f"Optimized theta = {result.optimized_theta}")
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
