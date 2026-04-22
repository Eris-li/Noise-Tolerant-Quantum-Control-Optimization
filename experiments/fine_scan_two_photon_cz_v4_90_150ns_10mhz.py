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
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    summarize_yb171_v4_result,
    yb171_gate_time_ns_to_dimensionless,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


OMEGA_MAX_HZ = 10e6
FINE_START_NS = 90.0
FINE_STOP_NS = 150.0
FINE_STEP_NS = 0.5
FINE_THRESHOLD = 0.999
FINE_NUM_TSLOTS = 200
FINE_ENSEMBLE_SIZE = 1
SEED = 17

MAX_ITER = 100
NUM_RESTARTS = 4
INIT_PULSE_TYPE = "SINE"
INIT_CONTROL_SCALE = 0.45
CONTROL_SMOOTHNESS_WEIGHT = 1e-3
CONTROL_CURVATURE_WEIGHT = 2e-3
AMPLITUDE_DIFF_WEIGHT = 50.0
PHASE_DIFF_WEIGHT = 15.0
AMPLITUDE_DIFF_THRESHOLD = 0.01
PHASE_DIFF_THRESHOLD = 0.1


def resample_controls(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == target_size:
        return np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.full(target_size, float(values[0]), dtype=np.float64)
    source_grid = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    target_grid = np.linspace(0.0, 1.0, target_size, dtype=np.float64)
    return np.asarray(np.interp(target_grid, source_grid, values), dtype=np.float64)


def build_config(gate_time_ns: float, seed: int) -> OpenSystemGRAPEConfig:
    return OpenSystemGRAPEConfig(
        num_tslots=FINE_NUM_TSLOTS,
        evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=OMEGA_MAX_HZ),
        max_iter=MAX_ITER,
        num_restarts=NUM_RESTARTS,
        seed=seed,
        init_pulse_type=INIT_PULSE_TYPE,
        init_control_scale=INIT_CONTROL_SCALE,
        control_smoothness_weight=CONTROL_SMOOTHNESS_WEIGHT,
        control_curvature_weight=CONTROL_CURVATURE_WEIGHT,
        amplitude_diff_weight=AMPLITUDE_DIFF_WEIGHT,
        phase_diff_weight=PHASE_DIFF_WEIGHT,
        amplitude_diff_threshold=AMPLITUDE_DIFF_THRESHOLD,
        phase_diff_threshold=PHASE_DIFF_THRESHOLD,
        fidelity_target=FINE_THRESHOLD,
        objective_metric="special_state",
        benchmark_active_channel=False,
        show_progress=True,
    )


def benchmark_runtime() -> dict[str, object]:
    times_ns = np.arange(FINE_START_NS, FINE_STOP_NS + 1e-12, FINE_STEP_NS, dtype=np.float64)
    return {
        "times_ns": [float(x) for x in times_ns],
        "num_points": int(times_ns.size),
        "num_tslots": FINE_NUM_TSLOTS,
        "max_iter": MAX_ITER,
        "num_restarts": NUM_RESTARTS,
        "init_pulse_type": INIT_PULSE_TYPE,
        "init_control_scale": INIT_CONTROL_SCALE,
        "control_smoothness_weight": CONTROL_SMOOTHNESS_WEIGHT,
        "control_curvature_weight": CONTROL_CURVATURE_WEIGHT,
        "amplitude_diff_weight": AMPLITUDE_DIFF_WEIGHT,
        "phase_diff_weight": PHASE_DIFF_WEIGHT,
        "amplitude_diff_threshold": AMPLITUDE_DIFF_THRESHOLD,
        "phase_diff_threshold": PHASE_DIFF_THRESHOLD,
    }


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    output_prefix = "two_photon_cz_v4_90_150ns_0p5ns_10mhz_fine"

    for stale_path in artifacts.glob(f"{output_prefix}*.json"):
        stale_path.unlink()

    print("[v4-fine] building nominal model and ensemble", flush=True)
    model = build_yb171_v4_calibrated_model(include_noise=True, effective_rabi_hz=OMEGA_MAX_HZ)
    ensemble_models = build_yb171_v4_quasistatic_ensemble(
        ensemble_size=FINE_ENSEMBLE_SIZE,
        seed=SEED,
        include_noise=True,
        effective_rabi_hz=OMEGA_MAX_HZ,
    )
    print("[v4-fine] model and ensemble ready", flush=True)

    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=build_config(FINE_START_NS, SEED),
        ensemble_models=ensemble_models,
    )

    fine_times_ns = [float(x) for x in np.arange(FINE_START_NS, FINE_STOP_NS + 1e-12, FINE_STEP_NS)]
    print(json.dumps({"fine_scan_config": benchmark_runtime()}, indent=2), flush=True)

    warm_start_ctrl_x: np.ndarray | None = None
    warm_start_ctrl_y: np.ndarray | None = None
    warm_start_theta = float(np.pi / 2.0)
    fine_points: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    threshold_first_ns: float | None = None

    scan_started = time.perf_counter()
    for index, gate_time_ns in enumerate(fine_times_ns, start=1):
        optimizer.reconfigure(build_config(gate_time_ns, SEED + index))
        started_at = time.perf_counter()
        print(f"[v4-fine] point {index}/{len(fine_times_ns)} : {gate_time_ns:.1f} ns", flush=True)
        result = optimizer.optimize(
            initial_ctrl_x=None if warm_start_ctrl_x is None else resample_controls(warm_start_ctrl_x, FINE_NUM_TSLOTS),
            initial_ctrl_y=None if warm_start_ctrl_y is None else resample_controls(warm_start_ctrl_y, FINE_NUM_TSLOTS),
            initial_theta=warm_start_theta,
        )
        summary = summarize_yb171_v4_result(
            result=result,
            gate_time_ns=gate_time_ns,
            omega_max_hz=OMEGA_MAX_HZ,
            model=optimizer.model,
        )
        summary["scan_stage"] = "fine"
        summary["ensemble_size"] = FINE_ENSEMBLE_SIZE
        summary["omega_mode"] = "yb171_default_10mhz"
        summary["wall_clock_scan_s"] = time.perf_counter() - started_at
        fine_points.append(summary)
        warm_start_ctrl_x = result.ctrl_x
        warm_start_ctrl_y = result.ctrl_y
        warm_start_theta = float(result.optimized_theta)
        if best_summary is None or float(summary["probe_fidelity"]) > float(best_summary["probe_fidelity"]):
            best_summary = summary
        if threshold_first_ns is None and float(summary["probe_fidelity"]) >= FINE_THRESHOLD:
            threshold_first_ns = gate_time_ns
        (artifacts / f"{output_prefix}_{gate_time_ns:.1f}ns.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - scan_started
        avg = elapsed / index
        eta = avg * (len(fine_times_ns) - index)
        print(
            f"[v4-fine] {gate_time_ns:.1f} ns F={summary['probe_fidelity']:.6f} "
            f"iter={summary['num_iter']} success={summary['success']} "
            f"elapsed={elapsed:7.1f}s eta={eta:7.1f}s",
            flush=True,
        )

    payload = {
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "omega_mode": "yb171_default_10mhz",
        "fine_scan_config": benchmark_runtime(),
        "fine_times_ns": fine_times_ns,
        "points": fine_points,
        "first_threshold_point": threshold_first_ns,
        "best_point_ns": None if best_summary is None else float(best_summary["gate_time_ns"]),
        "best_fidelity": 0.0 if best_summary is None else float(best_summary["probe_fidelity"]),
    }
    (artifacts / f"{output_prefix}_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
