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
    summarize_yb171_v4_result,
    yb171_dimensionless_time_to_gate_time_ns,
    yb171_gate_time_ns_to_dimensionless,
    yb171_v4_default_omega_max_hz,
)
from neutral_yb.optimization.open_system_grape import (
    OpenSystemGRAPEConfig,
    OpenSystemGRAPEOptimizer,
)


def main() -> None:
    omega_max_hz = yb171_v4_default_omega_max_hz()
    durations_dimensionless = [10.0, 9.0, 8.0, 7.5]
    durations_ns = [
        yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=omega_max_hz)
        for value in durations_dimensionless
    ]
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=yb171_gate_time_ns_to_dimensionless(durations_ns[0], effective_rabi_hz=omega_max_hz),
            max_iter=1,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=0.999,
            show_progress=True,
        ),
    )
    scan, results = optimizer.scan_durations(
        [
            yb171_gate_time_ns_to_dimensionless(value, effective_rabi_hz=omega_max_hz)
            for value in durations_ns
        ]
    )

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "two_photon_cz_v4_open_system_scout.json").write_text(
        json.dumps(
            {
                "gate_times_ns": durations_ns,
                "durations_dimensionless": scan.durations,
                "fidelities": scan.fidelities,
                "best_gate_time_ns": None
                if scan.best_duration is None
                else yb171_dimensionless_time_to_gate_time_ns(scan.best_duration, effective_rabi_hz=omega_max_hz),
                "best_duration_dimensionless": scan.best_duration,
                "best_fidelity": scan.best_fidelity,
                "target_reached": scan.target_reached,
                "omega_max_hz": omega_max_hz,
                "omega_max_mhz": omega_max_hz / 1e6,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "gate_times_ns": durations_ns,
        "durations_dimensionless": scan.durations,
        "fidelities": scan.fidelities,
        "best_fidelity": scan.best_fidelity,
        "best_gate_time_ns": None
        if scan.best_duration is None
        else yb171_dimensionless_time_to_gate_time_ns(scan.best_duration, effective_rabi_hz=omega_max_hz),
        "best_duration_dimensionless": scan.best_duration,
        "target_reached": scan.target_reached,
        "omega_max_hz": omega_max_hz,
        "omega_max_mhz": omega_max_hz / 1e6,
        "num_tslots": optimizer.config.num_tslots,
        "max_iter": optimizer.config.max_iter,
        "num_restarts": optimizer.config.num_restarts,
        "points": [
            summarize_yb171_v4_result(
                result=result,
                gate_time_ns=gate_time_ns,
                omega_max_hz=omega_max_hz,
                model=optimizer.model,
            )
            for gate_time_ns, result in zip(durations_ns, results)
        ],
    }
    destination = artifacts / "two_photon_cz_v4_open_system_scout_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Scout scan completed")
    print(f"Saved scan to {artifacts / 'two_photon_cz_v4_open_system_scout.json'}")
    print(f"Saved summary to {destination}")


if __name__ == "__main__":
    main()
