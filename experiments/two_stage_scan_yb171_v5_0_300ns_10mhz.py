from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ensure_artifact_dir, v5_coarse_10mhz_dir
from neutral_yb.config.yb171_calibration import (
    Yb171CalibrationProfile,
    build_yb171_v5_calibrated_model,
    build_yb171_v5_quasistatic_ensemble,
    summarize_yb171_v5_result,
    yb171_gate_time_ns_to_dimensionless,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


OMEGA_MAX_HZ = 10e6
COARSE_TIMES_NS = [float(value) for value in range(0, 301, 30)]
COARSE_THRESHOLD = 0.999
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["strict_literature_minimal", "experimental_surrogate_full"],
        required=True,
    )
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def profile_slug(profile: str) -> str:
    return profile.replace("-", "_")


def output_prefix(profile: str) -> str:
    return f"yb171_v5_{profile_slug(profile)}_0_300ns_10mhz"


def coarse_num_tslots(gate_time_ns: float) -> int:
    if gate_time_ns <= 0.0:
        return 1
    return 100


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
    seed: int,
) -> OpenSystemGRAPEConfig:
    return OpenSystemGRAPEConfig(
        num_tslots=num_tslots,
        evo_time=yb171_gate_time_ns_to_dimensionless(gate_time_ns, effective_rabi_hz=OMEGA_MAX_HZ),
        max_iter=COARSE_MAX_ITER,
        num_restarts=COARSE_NUM_RESTARTS,
        seed=seed,
        init_pulse_type=COARSE_INIT_PULSE_TYPE,
        init_control_scale=COARSE_INIT_CONTROL_SCALE,
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


def baseline_point() -> dict[str, object]:
    theta = float(np.pi / 2.0)
    fidelity = 0.6
    return {
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
        "model_version": "v5",
    }


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
    args = parse_args()
    profile: Yb171CalibrationProfile = args.profile
    artifacts = ensure_artifact_dir(v5_coarse_10mhz_dir(ROOT, profile))
    prefix = output_prefix(profile)

    for stale_path in artifacts.glob(f"{prefix}*.json"):
        stale_path.unlink()
    for stale_path in artifacts.glob(f"{prefix}*.png"):
        stale_path.unlink()

    print(f"[v5] building model and ensemble for profile={profile}", flush=True)
    model = build_yb171_v5_calibrated_model(
        include_noise=True,
        effective_rabi_hz=OMEGA_MAX_HZ,
        profile=profile,
    )
    ensemble_models = build_yb171_v5_quasistatic_ensemble(
        ensemble_size=max(int(args.ensemble_size), 1),
        seed=int(args.seed),
        include_noise=True,
        effective_rabi_hz=OMEGA_MAX_HZ,
        profile=profile,
    )
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=build_config(gate_time_ns=30.0, num_tslots=coarse_num_tslots(30.0), seed=int(args.seed)),
        ensemble_models=ensemble_models,
    )

    coarse_points: list[dict[str, object]] = [baseline_point()]
    previous_summary: dict[str, object] | None = None
    threshold_point: dict[str, object] | None = None
    total_started_at = time.perf_counter()

    for gate_time_ns in COARSE_TIMES_NS[1:]:
        print(f"[v5:{profile}] coarse point {gate_time_ns:.1f} ns", flush=True)
        optimizer.reconfigure(
            build_config(
                gate_time_ns=gate_time_ns,
                num_tslots=coarse_num_tslots(gate_time_ns),
                seed=int(args.seed),
            )
        )
        initial_ctrl_x = None
        initial_ctrl_y = None
        initial_theta = np.pi / 2.0
        if previous_summary is not None:
            initial_ctrl_x = resample_controls(
                np.asarray(previous_summary["ctrl_x"], dtype=np.float64),
                optimizer.config.num_tslots,
            )
            initial_ctrl_y = resample_controls(
                np.asarray(previous_summary["ctrl_y"], dtype=np.float64),
                optimizer.config.num_tslots,
            )
            initial_theta = float(previous_summary["optimized_theta"])
        started_at = time.perf_counter()
        result = optimizer.optimize(
            initial_ctrl_x=initial_ctrl_x,
            initial_ctrl_y=initial_ctrl_y,
            initial_theta=initial_theta,
        )
        summary = summarize_yb171_v5_result(
            result=result,
            gate_time_ns=gate_time_ns,
            omega_max_hz=OMEGA_MAX_HZ,
            model=optimizer.model,
        )
        summary["scan_stage"] = "coarse"
        summary["ensemble_size"] = int(args.ensemble_size)
        summary["profile_name"] = profile
        summary["omega_mode"] = "yb171_default_10mhz"
        summary["wall_clock_scan_s"] = time.perf_counter() - started_at
        previous_summary = summary
        coarse_points.append(summary)
        (artifacts / f"{prefix}_coarse_{gate_time_ns:.0f}ns.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(
            f"[v5:{profile}] coarse {gate_time_ns:.1f} ns F={summary['probe_fidelity']:.6f} "
            f"theta={summary['optimized_theta']:.6f}",
            flush=True,
        )
        if threshold_point is None and float(summary["probe_fidelity"]) >= COARSE_THRESHOLD:
            threshold_point = summary

    coarse_payload = {
        "model_version": "v5",
        "profile_name": profile,
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "coarse_times_ns": COARSE_TIMES_NS,
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
            "ensemble_size": int(args.ensemble_size),
        },
        "points": coarse_points,
        "first_threshold_point": None if threshold_point is None else float(threshold_point["gate_time_ns"]),
        "total_wall_clock_s": time.perf_counter() - total_started_at,
    }
    (artifacts / f"{prefix}_coarse_summary.json").write_text(
        json.dumps(coarse_payload, indent=2),
        encoding="utf-8",
    )

    has_problem, problem_reason = coarse_problem_detected(coarse_points[1:])
    final_payload = {
        "model_version": "v5",
        "profile_name": profile,
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "coarse_summary": coarse_payload,
        "fine_scan_started": False,
        "reason": (
            problem_reason
            if has_problem
            else ("no coarse point reached fidelity >= 0.999" if threshold_point is None else "coarse scan completed")
        ),
    }
    (artifacts / f"{prefix}_summary.json").write_text(
        json.dumps(final_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"profile_name": profile, "summary_path": str(artifacts / f'{prefix}_summary.json')}, indent=2))


if __name__ == "__main__":
    main()
