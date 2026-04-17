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


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def main() -> None:
    threshold = 0.995
    omega_max_hz = yb171_v4_default_omega_max_hz()
    durations_dimensionless = list(reversed(frange(1.0, 10.0, 1.0)))
    durations_ns = [
        yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=omega_max_hz)
        for value in durations_dimensionless
    ]
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=yb171_gate_time_ns_to_dimensionless(durations_ns[0], effective_rabi_hz=omega_max_hz),
            max_iter=5,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
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
    (artifacts / "two_photon_cz_v4_open_system_coarse.json").write_text(
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
                "points": [
                    summarize_yb171_v4_result(
                        result=result,
                        gate_time_ns=gate_time_ns,
                        omega_max_hz=omega_max_hz,
                        model=optimizer.model,
                    )
                    for gate_time_ns, result in zip(durations_ns, results)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best = max(results, key=lambda result: result.probe_fidelity)
    best_gate_time_ns = yb171_dimensionless_time_to_gate_time_ns(best.evo_time, effective_rabi_hz=omega_max_hz)
    (artifacts / "two_photon_cz_v4_open_system_best.json").write_text(
        json.dumps(
            summarize_yb171_v4_result(
                result=best,
                gate_time_ns=best_gate_time_ns,
                omega_max_hz=omega_max_hz,
                model=optimizer.model,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    qualifying = [result for result in results if result.probe_fidelity >= threshold]
    if qualifying:
        optimal = min(qualifying, key=lambda result: result.evo_time)
        optimal_gate_time_ns = yb171_dimensionless_time_to_gate_time_ns(optimal.evo_time, effective_rabi_hz=omega_max_hz)
        (artifacts / "two_photon_cz_v4_open_system_optimal.json").write_text(
            json.dumps(
                summarize_yb171_v4_result(
                    result=optimal,
                    gate_time_ns=optimal_gate_time_ns,
                    omega_max_hz=omega_max_hz,
                    model=optimizer.model,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Earliest threshold point = {optimal_gate_time_ns:.3f} ns")
        print(f"Threshold fidelity = {optimal.probe_fidelity}")
    else:
        print("No coarse-scan point reached the fidelity threshold")

    print("Two-photon CZ v4 open-system coarse scan completed")
    print(f"Best coarse gate time = {best_gate_time_ns:.3f} ns")
    print(f"Best coarse probe fidelity = {best.probe_fidelity}")
    print(f"Best coarse theta = {best.optimized_theta}")


if __name__ == "__main__":
    main()
