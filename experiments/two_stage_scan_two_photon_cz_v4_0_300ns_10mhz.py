from __future__ import annotations

from dataclasses import asdict, dataclass
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
    yb171_experimental_calibration,
)
from neutral_yb.config.artifact_paths import ensure_artifact_dir, v4_coarse_10mhz_dir
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


OMEGA_MAX_HZ = 10e6
COARSE_TIMES_NS = [float(value) for value in range(0, 301, 30)]
COARSE_THRESHOLD = 0.999
COARSE_ENSEMBLE_SIZE = 3
FINE_ENSEMBLE_SIZE = 3
SEED = 17
RUN_FINE_AFTER_COARSE = False
COARSE_MAX_ITER = 100
COARSE_NUM_RESTARTS = 4
COARSE_INIT_PULSE_TYPE = "SINE"
COARSE_INIT_CONTROL_SCALE = 0.45
COARSE_SMOOTHNESS_WEIGHT = 1e-3
COARSE_CURVATURE_WEIGHT = 2e-3
COARSE_AMPLITUDE_DIFF_WEIGHT = 50.0
COARSE_PHASE_DIFF_WEIGHT = 15.0
COARSE_AMPLITUDE_DIFF_THRESHOLD = 0.01
COARSE_PHASE_DIFF_THRESHOLD = 0.1


@dataclass(frozen=True)
class StageSpec:
    label: str
    max_iter: int
    num_restarts: int
    init_pulse_type: str
    init_control_scale: float


FINE_STAGES = [
    StageSpec("fine_stage1", max_iter=12, num_restarts=2, init_pulse_type="SINE", init_control_scale=0.35),
    StageSpec("fine_stage2", max_iter=24, num_restarts=3, init_pulse_type="SINE", init_control_scale=0.40),
    StageSpec("fine_stage3", max_iter=40, num_restarts=4, init_pulse_type="RANDOM", init_control_scale=0.50),
    StageSpec("fine_stage4", max_iter=60, num_restarts=4, init_pulse_type="SINE", init_control_scale=0.50),
]


def coarse_num_tslots(gate_time_ns: float) -> int:
    if gate_time_ns <= 0.0:
        return 1
    return 100


def fine_num_tslots(gate_time_ns: float) -> int:
    return max(20, int(round(gate_time_ns / 5.0)))


def resample_controls(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == target_size:
        return np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.full(target_size, float(values[0]), dtype=np.float64)
    source_grid = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    target_grid = np.linspace(0.0, 1.0, target_size, dtype=np.float64)
    return np.asarray(np.interp(target_grid, source_grid, values), dtype=np.float64)


def build_config(
    *,
    gate_time_ns: float,
    num_tslots: int,
    max_iter: int,
    num_restarts: int,
    ensemble_size: int,
    init_pulse_type: str,
    init_control_scale: float,
    seed: int,
) -> OpenSystemGRAPEConfig:
    _ = ensemble_size
    return OpenSystemGRAPEConfig(
        num_tslots=num_tslots,
        evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=OMEGA_MAX_HZ),
        max_iter=max_iter,
        num_restarts=num_restarts,
        seed=seed,
        init_pulse_type=init_pulse_type,
        init_control_scale=init_control_scale,
        control_smoothness_weight=COARSE_SMOOTHNESS_WEIGHT,
        control_curvature_weight=COARSE_CURVATURE_WEIGHT,
        amplitude_diff_weight=COARSE_AMPLITUDE_DIFF_WEIGHT,
        phase_diff_weight=COARSE_PHASE_DIFF_WEIGHT,
        amplitude_diff_threshold=COARSE_AMPLITUDE_DIFF_THRESHOLD,
        phase_diff_threshold=COARSE_PHASE_DIFF_THRESHOLD,
        fidelity_target=COARSE_THRESHOLD,
        objective_metric="special_state",
        benchmark_active_channel=True,
        show_progress=True,
    )


def baseline_point(model, ensemble_models: list[object]) -> dict[str, object]:
    theta = float(np.pi / 2.0)
    fidelity = 0.6
    payload = {
        "gate_time_ns": 0.0,
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "num_tslots": 1,
        "ctrl_x": [0.0],
        "ctrl_y": [0.0],
        "amplitudes": [0.0],
        "phases": [0.0],
        "optimized_theta": theta,
        "probe_fidelity": fidelity,
        "phase_gate_fidelity": fidelity,
        "fid_err": 1.0 - fidelity,
        "baseline_only": True,
    }
    return payload


def benchmark_runtime(model, ensemble_models: list[object]) -> dict[str, float]:
    return {
        "representative_gate_time_ns": 0.0,
        "coarse_point_wall_s": 0.0,
        "estimated_coarse_total_s": 0.0,
        "estimated_coarse_total_h": 0.0,
        "estimated_fine_total_s": 0.0,
        "estimated_fine_total_h": 0.0,
        "estimated_full_upper_bound_h": 0.0,
        "fine_estimation_method": "omitted",
        "coarse_config": {
            "max_iter": COARSE_MAX_ITER,
            "num_restarts": COARSE_NUM_RESTARTS,
            "init_pulse_type": COARSE_INIT_PULSE_TYPE,
            "init_control_scale": COARSE_INIT_CONTROL_SCALE,
            "control_smoothness_weight": COARSE_SMOOTHNESS_WEIGHT,
            "control_curvature_weight": COARSE_CURVATURE_WEIGHT,
            "amplitude_diff_weight": COARSE_AMPLITUDE_DIFF_WEIGHT,
            "phase_diff_weight": COARSE_PHASE_DIFF_WEIGHT,
            "amplitude_diff_threshold": COARSE_AMPLITUDE_DIFF_THRESHOLD,
            "phase_diff_threshold": COARSE_PHASE_DIFF_THRESHOLD,
        },
        "fine_stage_specs": [asdict(stage) for stage in FINE_STAGES],
    }


def coarse_optimize_point(
    optimizer: OpenSystemGRAPEOptimizer,
    gate_time_ns: float,
    previous_result: dict[str, object] | None,
) -> tuple[dict[str, object], object]:
    optimizer.reconfigure(
        build_config(
        gate_time_ns=gate_time_ns,
        num_tslots=coarse_num_tslots(gate_time_ns),
        max_iter=COARSE_MAX_ITER,
        num_restarts=COARSE_NUM_RESTARTS,
        ensemble_size=COARSE_ENSEMBLE_SIZE,
        init_pulse_type=COARSE_INIT_PULSE_TYPE,
        init_control_scale=COARSE_INIT_CONTROL_SCALE,
        seed=SEED,
        )
    )
    initial_ctrl_x = None
    initial_ctrl_y = None
    initial_theta = np.pi / 2.0
    if previous_result is not None:
        initial_ctrl_x = resample_controls(np.asarray(previous_result["ctrl_x"], dtype=np.float64), optimizer.config.num_tslots)
        initial_ctrl_y = resample_controls(np.asarray(previous_result["ctrl_y"], dtype=np.float64), optimizer.config.num_tslots)
        initial_theta = float(previous_result["optimized_theta"])
    result = optimizer.optimize(
        initial_ctrl_x=initial_ctrl_x,
        initial_ctrl_y=initial_ctrl_y,
        initial_theta=initial_theta,
    )
    summary = summarize_yb171_v4_result(
        result=result,
        gate_time_ns=gate_time_ns,
        omega_max_hz=OMEGA_MAX_HZ,
        model=optimizer.model,
    )
    summary["scan_stage"] = "coarse"
    summary["ensemble_size"] = COARSE_ENSEMBLE_SIZE
    summary["omega_mode"] = "yb171_default_10mhz"
    return summary, result


def fine_optimize_point(
    optimizer: OpenSystemGRAPEOptimizer,
    gate_time_ns: float,
    warm_start: dict[str, object],
) -> dict[str, object]:
    stage_records: list[dict[str, object]] = []
    current_ctrl_x = np.asarray(warm_start["ctrl_x"], dtype=np.float64)
    current_ctrl_y = np.asarray(warm_start["ctrl_y"], dtype=np.float64)
    current_theta = float(warm_start["optimized_theta"])
    best_summary: dict[str, object] | None = None

    for stage_index, stage in enumerate(FINE_STAGES):
        optimizer.reconfigure(
            build_config(
            gate_time_ns=gate_time_ns,
            num_tslots=fine_num_tslots(gate_time_ns),
            max_iter=stage.max_iter,
            num_restarts=stage.num_restarts,
            ensemble_size=FINE_ENSEMBLE_SIZE,
            init_pulse_type=stage.init_pulse_type,
            init_control_scale=stage.init_control_scale,
            seed=SEED + 100 + stage_index,
            )
        )
        result = optimizer.optimize(
            initial_ctrl_x=resample_controls(current_ctrl_x, optimizer.config.num_tslots),
            initial_ctrl_y=resample_controls(current_ctrl_y, optimizer.config.num_tslots),
            initial_theta=current_theta,
        )
        summary = summarize_yb171_v4_result(
            result=result,
            gate_time_ns=gate_time_ns,
            omega_max_hz=OMEGA_MAX_HZ,
            model=optimizer.model,
        )
        summary["fine_stage"] = stage.label
        stage_records.append(
            {
                "stage": stage.label,
                "max_iter": stage.max_iter,
                "num_restarts": stage.num_restarts,
                "probe_fidelity": summary["probe_fidelity"],
                "objective": summary["objective"],
                "num_iter": summary["num_iter"],
            }
        )
        if best_summary is None or float(summary["probe_fidelity"]) > float(best_summary["probe_fidelity"]):
            best_summary = summary
        current_ctrl_x = result.ctrl_x
        current_ctrl_y = result.ctrl_y
        current_theta = result.optimized_theta

        if float(summary["probe_fidelity"]) >= COARSE_THRESHOLD:
            break
        if len(stage_records) >= 2:
            improvement = stage_records[-1]["probe_fidelity"] - stage_records[-2]["probe_fidelity"]
            if improvement < 1e-4:
                break

    assert best_summary is not None
    best_summary["scan_stage"] = "fine"
    best_summary["ensemble_size"] = FINE_ENSEMBLE_SIZE
    best_summary["fine_stage_records"] = stage_records
    best_summary["omega_mode"] = "yb171_default_10mhz"
    return best_summary


def coarse_problem_detected(points: list[dict[str, object]]) -> tuple[bool, str]:
    if not points:
        return True, "coarse scan produced no optimized points"
    fidelities = [float(point["probe_fidelity"]) for point in points]
    if max(fidelities) < 0.80:
        return True, "coarse scan best fidelity stayed below 0.80"
    if sum(bool(point["success"]) for point in points) < max(1, len(points) // 3):
        return True, "too many coarse points failed optimizer termination"
    return False, ""


def main() -> None:
    artifacts = ensure_artifact_dir(v4_coarse_10mhz_dir(ROOT))
    output_prefix = "two_photon_cz_v4_0_300ns_10mhz"
    for stale_path in artifacts.glob(f"{output_prefix}*.json"):
        stale_path.unlink()
    for stale_path in artifacts.glob(f"{output_prefix}*.png"):
        stale_path.unlink()

    print("[v4-full] building nominal model and ensemble", flush=True)
    model = build_yb171_v4_calibrated_model(include_noise=True, effective_rabi_hz=OMEGA_MAX_HZ)
    ensemble_models = build_yb171_v4_quasistatic_ensemble(
        ensemble_size=max(COARSE_ENSEMBLE_SIZE, FINE_ENSEMBLE_SIZE),
        seed=SEED,
        include_noise=True,
        effective_rabi_hz=OMEGA_MAX_HZ,
    )
    print("[v4-full] model and ensemble ready", flush=True)
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=build_config(
            gate_time_ns=30.0,
            num_tslots=coarse_num_tslots(30.0),
            max_iter=COARSE_MAX_ITER,
            num_restarts=COARSE_NUM_RESTARTS,
            ensemble_size=COARSE_ENSEMBLE_SIZE,
            init_pulse_type=COARSE_INIT_PULSE_TYPE,
            init_control_scale=COARSE_INIT_CONTROL_SCALE,
            seed=SEED,
        ),
        ensemble_models=ensemble_models,
    )
    runtime_estimate = benchmark_runtime(model, ensemble_models)
    print(json.dumps({"runtime_estimate": runtime_estimate}, indent=2), flush=True)

    coarse_points: list[dict[str, object]] = [baseline_point(model, ensemble_models)]
    previous_summary: dict[str, object] | None = None
    threshold_point: dict[str, object] | None = None

    for gate_time_ns in COARSE_TIMES_NS[1:]:
        started_at = time.perf_counter()
        print(f"[v4-10mhz] coarse point {gate_time_ns:.1f} ns", flush=True)
        summary, _result = coarse_optimize_point(optimizer, gate_time_ns, previous_summary)
        summary["wall_clock_scan_s"] = time.perf_counter() - started_at
        coarse_points.append(summary)
        previous_summary = summary
        (artifacts / f"{output_prefix}_coarse_{gate_time_ns:.0f}ns.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(
            f"[v4-10mhz] coarse {gate_time_ns:.1f} ns F={summary['probe_fidelity']:.6f} "
            f"theta={summary['optimized_theta']:.6f}",
            flush=True,
        )
        if threshold_point is None and float(summary["probe_fidelity"]) >= COARSE_THRESHOLD:
            threshold_point = summary
            if RUN_FINE_AFTER_COARSE:
                break

    coarse_payload = {
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "omega_mode": "yb171_default_10mhz",
        "coarse_times_ns": COARSE_TIMES_NS,
        "runtime_estimate": runtime_estimate,
        "points": coarse_points,
        "first_threshold_point": None if threshold_point is None else float(threshold_point["gate_time_ns"]),
    }
    (artifacts / f"{output_prefix}_coarse_summary.json").write_text(
        json.dumps(coarse_payload, indent=2),
        encoding="utf-8",
    )

    has_problem, problem_reason = coarse_problem_detected(coarse_points[1:])
    if threshold_point is None or has_problem or not RUN_FINE_AFTER_COARSE:
        final_payload = {
            "omega_max_hz": OMEGA_MAX_HZ,
            "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
            "omega_mode": "yb171_default_10mhz",
            "runtime_estimate": runtime_estimate,
            "coarse_summary": coarse_payload,
            "fine_scan_started": False,
            "reason": (
                problem_reason
                if has_problem
                else ("fine scan disabled for this run" if RUN_FINE_AFTER_COARSE is False else "no coarse point reached fidelity >= 0.999")
            ),
        }
        (artifacts / f"{output_prefix}_summary.json").write_text(
            json.dumps(final_payload, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(final_payload, indent=2), flush=True)
        return

    threshold_time_ns = float(threshold_point["gate_time_ns"])
    fine_start_ns = max(0.0, threshold_time_ns - 30.0)
    fine_times_ns = [float(value) for value in np.arange(fine_start_ns, threshold_time_ns + 1e-12, 1.0)]
    fine_points: list[dict[str, object]] = []
    warm_start = threshold_point
    previous_coarse = [point for point in coarse_points if float(point["gate_time_ns"]) < threshold_time_ns and "ctrl_x" in point]
    if previous_coarse:
        warm_start = previous_coarse[-1]

    for gate_time_ns in fine_times_ns:
        if gate_time_ns == 0.0:
            fine_points.append(baseline_point(model, ensemble_models))
            continue
        started_at = time.perf_counter()
        print(f"[v4-10mhz] fine point {gate_time_ns:.1f} ns", flush=True)
        summary = fine_optimize_point(optimizer, gate_time_ns, warm_start)
        summary["wall_clock_scan_s"] = time.perf_counter() - started_at
        fine_points.append(summary)
        warm_start = summary
        (artifacts / f"{output_prefix}_fine_{gate_time_ns:.0f}ns.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(
            f"[v4-10mhz] fine {gate_time_ns:.1f} ns F={summary['probe_fidelity']:.6f} "
            f"stages={len(summary['fine_stage_records'])}",
            flush=True,
        )

    fine_payload = {
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "omega_mode": "yb171_default_10mhz",
        "fine_times_ns": fine_times_ns,
        "points": fine_points,
    }
    (artifacts / f"{output_prefix}_fine_summary.json").write_text(
        json.dumps(fine_payload, indent=2),
        encoding="utf-8",
    )

    final_payload = {
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "omega_mode": "yb171_default_10mhz",
        "runtime_estimate": runtime_estimate,
        "coarse_summary": coarse_payload,
        "fine_summary": fine_payload,
        "fine_scan_started": True,
    }
    (artifacts / f"{output_prefix}_summary.json").write_text(
        json.dumps(final_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
