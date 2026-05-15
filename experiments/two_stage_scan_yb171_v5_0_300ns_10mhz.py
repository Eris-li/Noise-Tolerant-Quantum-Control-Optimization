from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ensure_artifact_dir, v5_profile_dir
from neutral_yb.config.yb171_calibration import (
    Yb171CalibrationProfile,
    build_yb171_v5_calibrated_model,
    build_yb171_v5_quasistatic_ensemble,
    summarize_yb171_v5_result,
    yb171_experimental_calibration,
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
COARSE_RADIAL_AMPLITUDE_BOUND_WEIGHT = 1000.0
COARSE_AMPLITUDE_DIFF_THRESHOLD = 0.01
COARSE_PHASE_DIFF_THRESHOLD = 0.1
UV_CONTROL_ENVELOPE = "GAUSSIAN_EDGE"
UV_GAUSSIAN_EDGE_FRACTION = 0.20
UV_GAUSSIAN_EDGE_SIGMA_FRACTION = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["strict_literature_minimal", "experimental_surrogate_full"],
        required=True,
    )
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def profile_slug(profile: str) -> str:
    return profile.replace("-", "_")


def output_prefix(profile: str) -> str:
    return f"yb171_v5_{profile_slug(profile)}_0_300ns_10mhz"


def resolve_run_label(value: str | None) -> str:
    if value is not None and value.strip():
        return value.strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        radial_amplitude_bound_weight=COARSE_RADIAL_AMPLITUDE_BOUND_WEIGHT,
        amplitude_diff_threshold=COARSE_AMPLITUDE_DIFF_THRESHOLD,
        phase_diff_threshold=COARSE_PHASE_DIFF_THRESHOLD,
        control_envelope=UV_CONTROL_ENVELOPE,
        gaussian_edge_fraction=UV_GAUSSIAN_EDGE_FRACTION,
        gaussian_edge_sigma_fraction=UV_GAUSSIAN_EDGE_SIGMA_FRACTION,
        fidelity_target=COARSE_THRESHOLD,
        objective_metric="active_channel",
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
        "raw_ctrl_x": [0.0],
        "raw_ctrl_y": [0.0],
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


def finite_float_or_none(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def noise_config_payload(noise) -> dict[str, object]:
    payload = asdict(noise)
    return {
        key: ([float(item) for item in value] if isinstance(value, tuple) else finite_float_or_none(value))
        for key, value in payload.items()
    }


def physical_system_parameters(
    *,
    profile: Yb171CalibrationProfile,
    model,
    ensemble_models: list,
) -> dict[str, object]:
    calibration = yb171_experimental_calibration(profile=profile)
    calibration_summary = calibration.summary(effective_rabi_hz=OMEGA_MAX_HZ)
    resolved_uv_detuning_hz = float(calibration_summary["resolved_quasistatic_uv_detuning_rms_hz"])
    return {
        "model_version": "v5",
        "model_kind": calibration_summary["model_kind"],
        "profile_name": profile,
        "basis_labels": list(model.basis_labels()),
        "active_gate_indices": list(model.active_gate_indices()),
        "drive_parameters": {
            "omega_max_hz": OMEGA_MAX_HZ,
            "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
            "uv_rabi_hz": float(calibration.uv_rabi_hz),
            "uv_rabi_mhz": float(calibration.uv_rabi_hz / 1e6),
            "uv_rabi_dimensionless": finite_float_or_none(calibration_summary["uv_rabi_dimensionless"]),
            "clock_shelving_rabi_hz": float(calibration.clock_shelving_rabi_hz),
            "clock_pi_pulse_duration_s": float(calibration.clock_pi_pulse_duration_s),
            "clock_pi_pulse_duration_us": float(calibration_summary["clock_pi_pulse_duration_us"]),
            "clock_num_steps": int(calibration.clock_num_steps),
            "fixed_prefix_duration_us": float(calibration_summary["clock_total_duration_us"]),
            "fixed_suffix_duration_us": float(calibration_summary["clock_total_duration_us"]),
            "blockade_shift_hz": float(calibration.blockade_shift_hz),
            "blockade_shift_mhz": float(calibration.blockade_shift_hz / 1e6),
            "blockade_shift_dimensionless": finite_float_or_none(
                calibration_summary["blockade_shift_dimensionless"]
            ),
            "rydberg_state_label": calibration.rydberg_state_label,
            "clock_state_label": calibration.clock_state_label,
        },
        "detuning_parameters": {
            "static_clock_detuning_01_dimensionless": float(model.static_clock_detuning_01),
            "static_clock_detuning_11_dimensionless": float(model.static_clock_detuning_11),
            "static_uv_detuning_01_dimensionless": float(model.static_uv_detuning_01),
            "static_uv_detuning_11_dimensionless": float(model.static_uv_detuning_11),
            "quasistatic_clock_detuning_rms_hz": float(calibration.quasistatic_clock_detuning_rms_hz),
            "differential_clock_detuning_rms_hz": float(calibration.differential_clock_detuning_rms_hz),
            "quasistatic_uv_detuning_rms_hz": resolved_uv_detuning_hz,
            "quasistatic_uv_detuning_rms_khz": resolved_uv_detuning_hz / 1e3,
            "quasistatic_uv_detuning_dimensionless": finite_float_or_none(
                calibration_summary["resolved_quasistatic_uv_detuning_dimensionless"]
            ),
            "differential_uv_detuning_rms_hz": float(calibration.differential_uv_detuning_rms_hz),
            "blockade_shift_jitter_hz": float(calibration.blockade_shift_jitter_hz),
        },
        "noise_parameters": {
            "lifetimes_s": {
                "clock_state": finite_float_or_none(calibration.clock_state_lifetime_s),
                "clock_trap_loss": finite_float_or_none(calibration.clock_trap_loss_lifetime_s),
                "rydberg": finite_float_or_none(calibration.rydberg_lifetime_s),
            },
            "coherence_s": {
                "rydberg_t2_star": finite_float_or_none(calibration.rydberg_t2_star_s),
                "rydberg_t2_echo": finite_float_or_none(calibration.rydberg_t2_echo_s),
                "derived_rydberg_pure_dephasing_time": finite_float_or_none(
                    calibration_summary["derived_rydberg_pure_dephasing_time_s"]
                ),
            },
            "markovian_rates_dimensionless": {
                "clock_decay_rate": finite_float_or_none(model.noise.clock_decay_rate),
                "clock_scattering_rate": finite_float_or_none(model.noise.clock_scattering_rate),
                "clock_loss_rate": finite_float_or_none(model.noise.clock_loss_rate),
                "clock_dephasing_rate": finite_float_or_none(model.noise.clock_dephasing_rate),
                "rydberg_decay_rate": finite_float_or_none(model.noise.rydberg_decay_rate),
                "rydberg_dephasing_rate": finite_float_or_none(model.noise.rydberg_dephasing_rate),
                "neighboring_mf_leakage_rate": finite_float_or_none(model.noise.neighboring_mf_leakage_rate),
            },
            "quasistatic_rms": {
                "clock_temperature_nbar": float(calibration.clock_temperature_nbar),
                "clock_lamb_dicke_eta": float(calibration.clock_lamb_dicke_eta),
                "clock_pulse_area_fractional_rms": float(calibration.clock_pulse_area_fractional_rms),
                "uv_pulse_area_fractional_rms": float(calibration.uv_pulse_area_fractional_rms),
                "clock_detuning_rms_hz": float(calibration.quasistatic_clock_detuning_rms_hz),
                "uv_detuning_rms_hz": resolved_uv_detuning_hz,
                "blockade_shift_jitter_hz": float(calibration.blockade_shift_jitter_hz),
            },
            "clock_phase_noise": {
                "psd_fmin_hz": float(calibration.clock_phase_noise_psd_fmin_hz),
                "psd_fmax_hz": float(calibration.clock_phase_noise_psd_fmax_hz),
                "num_bins": int(calibration.clock_phase_noise_num_bins),
                "psd_level_rad2_per_hz": float(calibration.clock_phase_noise_psd_level_rad2_per_hz),
            },
            "nominal_noise_config": noise_config_payload(model.noise),
            "ensemble_noise_realizations": [
                noise_config_payload(ensemble_model.noise) for ensemble_model in ensemble_models
            ],
        },
    }


def optimization_parameters(*, ensemble_size: int, seed: int) -> dict[str, object]:
    return {
        "scan_stage": "coarse",
        "coarse_times_ns": COARSE_TIMES_NS,
        "target_fidelity": COARSE_THRESHOLD,
        "ensemble_size": int(ensemble_size),
        "seed": int(seed),
        "num_tslots_rule": {
            "baseline_tslots": 1,
            "optimized_point_tslots": 100,
        },
        "grape_config": {
            "max_iter": COARSE_MAX_ITER,
            "num_restarts": COARSE_NUM_RESTARTS,
            "init_pulse_type": COARSE_INIT_PULSE_TYPE,
            "init_control_scale": COARSE_INIT_CONTROL_SCALE,
            "objective_metric": "active_channel",
            "benchmark_active_channel": True,
            "control_smoothness_weight": COARSE_SMOOTHNESS_WEIGHT,
            "control_curvature_weight": COARSE_CURVATURE_WEIGHT,
            "amplitude_diff_weight": COARSE_AMPLITUDE_DIFF_WEIGHT,
            "phase_diff_weight": COARSE_PHASE_DIFF_WEIGHT,
            "radial_amplitude_bound_weight": COARSE_RADIAL_AMPLITUDE_BOUND_WEIGHT,
            "amplitude_diff_threshold": COARSE_AMPLITUDE_DIFF_THRESHOLD,
            "phase_diff_threshold": COARSE_PHASE_DIFF_THRESHOLD,
            "control_envelope": UV_CONTROL_ENVELOPE,
            "gaussian_edge_fraction": UV_GAUSSIAN_EDGE_FRACTION,
            "gaussian_edge_sigma_fraction": UV_GAUSSIAN_EDGE_SIGMA_FRACTION,
        },
    }


def compact_point(point: dict[str, object]) -> dict[str, object]:
    return {
        "gate_time_ns": float(point["gate_time_ns"]),
        "uv_segment_time_ns": float(point.get("uv_segment_time_ns", point["gate_time_ns"])),
        "total_gate_time_us": (
            None if point.get("total_gate_time_us") is None else float(point["total_gate_time_us"])
        ),
        "probe_fidelity": float(point["probe_fidelity"]),
        "active_channel_fidelity": (
            None if point.get("active_channel_fidelity") is None else float(point["active_channel_fidelity"])
        ),
        "objective_fidelity": float(point.get("objective_fidelity", point["probe_fidelity"])),
        "objective_metric": point.get("objective_metric"),
        "optimized_theta": float(point["optimized_theta"]),
        "fid_err": float(point["fid_err"]),
        "success": None if point.get("success") is None else bool(point["success"]),
        "termination_reason": point.get("termination_reason"),
        "num_iter": point.get("num_iter"),
        "num_fid_func_calls": point.get("num_fid_func_calls"),
        "baseline_only": bool(point.get("baseline_only", False)),
    }


def best_point(points: list[dict[str, object]]) -> dict[str, object]:
    return max(points, key=lambda point: float(point["probe_fidelity"]))


def optimization_summary_payload(
    *,
    points: list[dict[str, object]],
    threshold_point: dict[str, object] | None,
    has_problem: bool,
    reason: str,
    total_wall_clock_s: float,
) -> dict[str, object]:
    best = best_point(points)
    return {
        "best_point": compact_point(best),
        "first_threshold_point": None if threshold_point is None else compact_point(threshold_point),
        "target_reached": threshold_point is not None,
        "fine_scan_started": False,
        "status": "problem_detected" if has_problem else "coarse_scan_completed",
        "reason": reason,
        "total_wall_clock_s": float(total_wall_clock_s),
    }


def detailed_results_payload(*, points: list[dict[str, object]]) -> dict[str, object]:
    best = best_point(points)
    return {
        "scan_points": [compact_point(point) for point in points],
        "best_control_sequences": {
            "gate_time_ns": float(best["gate_time_ns"]),
            "num_tslots": int(best["num_tslots"]),
            "slot_midpoints_ns": best.get("slot_midpoints_ns", []),
            "ctrl_x": best.get("ctrl_x", []),
            "ctrl_y": best.get("ctrl_y", []),
            "raw_ctrl_x": best.get("raw_ctrl_x", []),
            "raw_ctrl_y": best.get("raw_ctrl_y", []),
            "amplitudes": best.get("amplitudes", []),
            "phases": best.get("phases", []),
            "effective_rabi_fraction": best.get("effective_rabi_fraction", []),
            "effective_rabi_sequence_hz": best.get("effective_rabi_sequence_hz", []),
            "effective_rabi_sequence_mhz": best.get("effective_rabi_sequence_mhz", []),
        },
        "all_point_control_sequences": [
            {
                "gate_time_ns": float(point["gate_time_ns"]),
                "num_tslots": int(point["num_tslots"]),
                "slot_midpoints_ns": point.get("slot_midpoints_ns", []),
                "ctrl_x": point.get("ctrl_x", []),
                "ctrl_y": point.get("ctrl_y", []),
                "raw_ctrl_x": point.get("raw_ctrl_x", []),
                "raw_ctrl_y": point.get("raw_ctrl_y", []),
                "amplitudes": point.get("amplitudes", []),
                "phases": point.get("phases", []),
                "effective_rabi_fraction": point.get("effective_rabi_fraction", []),
                "effective_rabi_sequence_mhz": point.get("effective_rabi_sequence_mhz", []),
            }
            for point in points
        ],
    }


def main() -> None:
    args = parse_args()
    profile: Yb171CalibrationProfile = args.profile
    run_label = resolve_run_label(args.run_label)
    artifacts = ensure_artifact_dir(v5_profile_dir(ROOT, profile) / run_label / "coarse_0_300ns_10mhz")
    prefix = output_prefix(profile)

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
                np.asarray(previous_summary.get("raw_ctrl_x", previous_summary["ctrl_x"]), dtype=np.float64),
                optimizer.config.num_tslots,
            )
            initial_ctrl_y = resample_controls(
                np.asarray(previous_summary.get("raw_ctrl_y", previous_summary["ctrl_y"]), dtype=np.float64),
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

    total_wall_clock_s = time.perf_counter() - total_started_at
    physical_payload = physical_system_parameters(
        profile=profile,
        model=model,
        ensemble_models=ensemble_models,
    )
    opt_payload = optimization_parameters(ensemble_size=int(args.ensemble_size), seed=int(args.seed))
    has_problem, problem_reason = coarse_problem_detected(coarse_points[1:])
    reason = (
        problem_reason
        if has_problem
        else ("no coarse point reached fidelity >= 0.999" if threshold_point is None else "coarse scan completed")
    )
    summary_payload = optimization_summary_payload(
        points=coarse_points,
        threshold_point=threshold_point,
        has_problem=has_problem,
        reason=reason,
        total_wall_clock_s=total_wall_clock_s,
    )
    details_payload = detailed_results_payload(points=coarse_points)

    coarse_payload = {
        "physical_system_parameters": physical_payload,
        "optimization_parameters": opt_payload,
        "optimization_summary": summary_payload,
        "model_version": "v5",
        "profile_name": profile,
        "run_label": run_label,
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
            "radial_amplitude_bound_weight": COARSE_RADIAL_AMPLITUDE_BOUND_WEIGHT,
            "amplitude_diff_threshold": COARSE_AMPLITUDE_DIFF_THRESHOLD,
            "phase_diff_threshold": COARSE_PHASE_DIFF_THRESHOLD,
            "control_envelope": UV_CONTROL_ENVELOPE,
            "gaussian_edge_fraction": UV_GAUSSIAN_EDGE_FRACTION,
            "gaussian_edge_sigma_fraction": UV_GAUSSIAN_EDGE_SIGMA_FRACTION,
            "ensemble_size": int(args.ensemble_size),
        },
        "points": coarse_points,
        "first_threshold_point": None if threshold_point is None else float(threshold_point["gate_time_ns"]),
        "total_wall_clock_s": total_wall_clock_s,
        "detailed_results": details_payload,
    }
    (artifacts / f"{prefix}_coarse_summary.json").write_text(
        json.dumps(coarse_payload, indent=2),
        encoding="utf-8",
    )

    final_payload = {
        "physical_system_parameters": physical_payload,
        "optimization_parameters": opt_payload,
        "optimization_summary": summary_payload,
        "model_version": "v5",
        "profile_name": profile,
        "run_label": run_label,
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "coarse_summary": coarse_payload,
        "fine_scan_started": False,
        "reason": reason,
        "detailed_results": details_payload,
    }
    (artifacts / f"{prefix}_summary.json").write_text(
        json.dumps(final_payload, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "profile_name": profile,
                "run_label": run_label,
                "summary_path": str(artifacts / f"{prefix}_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
