from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy.optimize import curve_fit

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


def fit_time_optimal(durations: list[float], fidelities: list[float]) -> dict[str, float]:
    times = np.asarray(durations, dtype=float)
    errors = 1.0 - np.asarray(fidelities, dtype=float)
    mask = np.asarray(fidelities, dtype=float) >= 0.995
    if int(np.sum(mask)) < 3:
        mask = np.asarray(fidelities, dtype=float) >= 0.99
    t_fit = times[mask]
    e_fit = errors[mask]
    if t_fit.size < 3:
        raise RuntimeError("Not enough high-fidelity fine-scan points for a stable quadratic fit")

    def model(t: np.ndarray, t_star: float, scale: float) -> np.ndarray:
        return scale * np.maximum(t_star - t, 0.0) ** 2

    popt, _ = curve_fit(
        model,
        t_fit,
        e_fit,
        p0=[float(np.min(t_fit)), 0.05],
        bounds=([float(np.min(t_fit)) - 0.2, 0.0], [float(np.max(t_fit)) + 0.2, 10.0]),
    )
    return {"t_star": float(popt[0]), "scale": float(popt[1])}


def main() -> None:
    threshold = 0.999
    omega_max_hz = yb171_v4_default_omega_max_hz()
    coarse_durations_dimensionless = list(reversed(frange(0.5, 10.0, 0.5)))
    coarse_gate_times_ns = [
        yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=omega_max_hz)
        for value in coarse_durations_dimensionless
    ]
    coarse_optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=32,
            evo_time=yb171_gate_time_ns_to_dimensionless(coarse_gate_times_ns[0], effective_rabi_hz=omega_max_hz),
            max_iter=3,
            num_restarts=2,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
            show_progress=True,
        ),
    )
    coarse_scan, coarse_results = coarse_optimizer.scan_durations(
        [
            yb171_gate_time_ns_to_dimensionless(value, effective_rabi_hz=omega_max_hz)
            for value in coarse_gate_times_ns
        ]
    )

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "two_photon_cz_v4_open_system_coarse.json").write_text(
        json.dumps(
            {
                "gate_times_ns": coarse_gate_times_ns,
                "durations_dimensionless": coarse_scan.durations,
                "fidelities": coarse_scan.fidelities,
                "best_gate_time_ns": None
                if coarse_scan.best_duration is None
                else yb171_dimensionless_time_to_gate_time_ns(
                    coarse_scan.best_duration,
                    effective_rabi_hz=omega_max_hz,
                ),
                "best_duration_dimensionless": coarse_scan.best_duration,
                "best_fidelity": coarse_scan.best_fidelity,
                "target_reached": coarse_scan.target_reached,
                "omega_max_hz": omega_max_hz,
                "omega_max_mhz": omega_max_hz / 1e6,
                "points": [
                    summarize_yb171_v4_result(
                        result=result,
                        gate_time_ns=gate_time_ns,
                        omega_max_hz=omega_max_hz,
                        model=coarse_optimizer.model,
                    )
                    for gate_time_ns, result in zip(coarse_gate_times_ns, coarse_results)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    coarse_qualifying = [res for res in coarse_results if res.probe_fidelity >= threshold]
    if not coarse_qualifying:
        raise RuntimeError("No coarse-scan point reached fidelity >= 0.999")

    coarse_threshold_point = min(coarse_qualifying, key=lambda res: res.evo_time)
    fine_durations_dimensionless = list(reversed(frange(7.5, coarse_threshold_point.evo_time, 0.025)))
    fine_gate_times_ns = [
        yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=omega_max_hz)
        for value in fine_durations_dimensionless
    ]

    fine_optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=100,
            evo_time=yb171_gate_time_ns_to_dimensionless(fine_gate_times_ns[0], effective_rabi_hz=omega_max_hz),
            max_iter=5,
            num_restarts=10,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
            show_progress=True,
        ),
    )
    fine_scan, fine_results = fine_optimizer.scan_durations(
        [
            yb171_gate_time_ns_to_dimensionless(value, effective_rabi_hz=omega_max_hz)
            for value in fine_gate_times_ns
        ],
        initial_ctrl_x=coarse_threshold_point.ctrl_x,
        initial_ctrl_y=coarse_threshold_point.ctrl_y,
        initial_theta=coarse_threshold_point.optimized_theta,
    )
    (artifacts / "two_photon_cz_v4_open_system_fine.json").write_text(
        json.dumps(
            {
                "gate_times_ns": fine_gate_times_ns,
                "durations_dimensionless": fine_scan.durations,
                "fidelities": fine_scan.fidelities,
                "best_gate_time_ns": None
                if fine_scan.best_duration is None
                else yb171_dimensionless_time_to_gate_time_ns(
                    fine_scan.best_duration,
                    effective_rabi_hz=omega_max_hz,
                ),
                "best_duration_dimensionless": fine_scan.best_duration,
                "best_fidelity": fine_scan.best_fidelity,
                "target_reached": fine_scan.target_reached,
                "omega_max_hz": omega_max_hz,
                "omega_max_mhz": omega_max_hz / 1e6,
                "points": [
                    summarize_yb171_v4_result(
                        result=result,
                        gate_time_ns=gate_time_ns,
                        omega_max_hz=omega_max_hz,
                        model=fine_optimizer.model,
                    )
                    for gate_time_ns, result in zip(fine_gate_times_ns, fine_results)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fine_qualifying = [res for res in fine_results if res.probe_fidelity >= threshold]
    if not fine_qualifying:
        raise RuntimeError("No fine-scan point reached fidelity >= 0.999")
    optimal = min(fine_qualifying, key=lambda res: res.evo_time)
    optimal_gate_time_ns = yb171_dimensionless_time_to_gate_time_ns(optimal.evo_time, effective_rabi_hz=omega_max_hz)
    (artifacts / "two_photon_cz_v4_open_system_optimal.json").write_text(
        json.dumps(
            summarize_yb171_v4_result(
                result=optimal,
                gate_time_ns=optimal_gate_time_ns,
                omega_max_hz=omega_max_hz,
                model=fine_optimizer.model,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    best = max(fine_results, key=lambda result: result.probe_fidelity)
    best_gate_time_ns = yb171_dimensionless_time_to_gate_time_ns(best.evo_time, effective_rabi_hz=omega_max_hz)
    (artifacts / "two_photon_cz_v4_open_system_best.json").write_text(
        json.dumps(
            summarize_yb171_v4_result(
                result=best,
                gate_time_ns=best_gate_time_ns,
                omega_max_hz=omega_max_hz,
                model=fine_optimizer.model,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    fit = fit_time_optimal(fine_gate_times_ns, fine_scan.fidelities)
    fit["t_star_dimensionless"] = yb171_gate_time_ns_to_dimensionless(fit["t_star"], effective_rabi_hz=omega_max_hz)
    (artifacts / "two_photon_cz_v4_open_system_fit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

    print("Two-stage v4 open-system scan completed")
    print(
        "Coarse first threshold point = "
        f"{yb171_dimensionless_time_to_gate_time_ns(coarse_threshold_point.evo_time, effective_rabi_hz=omega_max_hz):.3f} ns"
    )
    print(f"Fine optimal gate time = {optimal_gate_time_ns:.3f} ns")
    print(f"Fine optimal fidelity = {optimal.probe_fidelity}")
    print(f"Fitted gate time = {fit['t_star']:.3f} ns")


if __name__ == "__main__":
    main()
