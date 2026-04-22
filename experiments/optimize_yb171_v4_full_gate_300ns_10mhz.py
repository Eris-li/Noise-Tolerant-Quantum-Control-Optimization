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
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


OMEGA_MAX_HZ = 10e6
GATE_TIME_NS = 300.0
ENSEMBLE_SIZE = 1
SEED = 17


@dataclass(frozen=True)
class StageSpec:
    label: str
    num_tslots: int
    max_iter: int
    num_restarts: int
    init_pulse_type: str
    init_control_scale: float
    control_smoothness_weight: float
    control_curvature_weight: float


STAGES = [
    StageSpec(
        label="sine_search",
        num_tslots=60,
        max_iter=80,
        num_restarts=4,
        init_pulse_type="SINE",
        init_control_scale=0.45,
        control_smoothness_weight=0.0,
        control_curvature_weight=0.0,
    ),
    StageSpec(
        label="sine_refine",
        num_tslots=100,
        max_iter=140,
        num_restarts=6,
        init_pulse_type="SINE",
        init_control_scale=0.55,
        control_smoothness_weight=0.0,
        control_curvature_weight=0.0,
    ),
    StageSpec(
        label="final_refine",
        num_tslots=140,
        max_iter=220,
        num_restarts=6,
        init_pulse_type="SINE",
        init_control_scale=0.55,
        control_smoothness_weight=0.0,
        control_curvature_weight=0.0,
    ),
]


def build_config(stage: StageSpec, *, seed: int) -> OpenSystemGRAPEConfig:
    return OpenSystemGRAPEConfig(
        num_tslots=stage.num_tslots,
        evo_time=yb171_gate_time_ns_to_dimensionless(GATE_TIME_NS, effective_rabi_hz=OMEGA_MAX_HZ),
        max_iter=stage.max_iter,
        num_restarts=stage.num_restarts,
        seed=seed,
        init_pulse_type=stage.init_pulse_type,
        init_control_scale=stage.init_control_scale,
        control_smoothness_weight=stage.control_smoothness_weight,
        control_curvature_weight=stage.control_curvature_weight,
        fidelity_target=0.999,
        objective_metric="special_state",
        benchmark_active_channel=False,
        show_progress=True,
    )


def resample_controls(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == target_size:
        return np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.full(target_size, float(values[0]), dtype=np.float64)
    source_grid = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    target_grid = np.linspace(0.0, 1.0, target_size, dtype=np.float64)
    return np.asarray(np.interp(target_grid, source_grid, values), dtype=np.float64)


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts / "two_photon_cz_v4_full_gate_300ns_10mhz_summary.json"
    result_path = artifacts / "two_photon_cz_v4_full_gate_300ns_10mhz_best.json"

    model = build_yb171_v4_calibrated_model(include_noise=True, effective_rabi_hz=OMEGA_MAX_HZ)
    ensemble_models = build_yb171_v4_quasistatic_ensemble(
        ensemble_size=ENSEMBLE_SIZE,
        seed=SEED,
        include_noise=True,
        effective_rabi_hz=OMEGA_MAX_HZ,
    )
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=build_config(STAGES[0], seed=SEED),
        ensemble_models=ensemble_models,
    )

    baseline = summarize_yb171_v4_result(
        result=optimizer._zero_control_baseline_result(),
        gate_time_ns=GATE_TIME_NS,
        omega_max_hz=OMEGA_MAX_HZ,
        model=model,
    )
    baseline["stage"] = "zero_baseline"

    current_ctrl_x = None
    current_ctrl_y = None
    current_theta = np.pi / 2.0
    best_summary = baseline
    best_result = None
    stage_summaries: list[dict[str, object]] = [baseline]

    for index, stage in enumerate(STAGES):
        optimizer.reconfigure(build_config(stage, seed=SEED + 100 * (index + 1)))
        started_at = time.perf_counter()
        result = optimizer.optimize(
            initial_ctrl_x=None if current_ctrl_x is None else resample_controls(current_ctrl_x, optimizer.config.num_tslots),
            initial_ctrl_y=None if current_ctrl_y is None else resample_controls(current_ctrl_y, optimizer.config.num_tslots),
            initial_theta=current_theta,
        )
        wall_clock = time.perf_counter() - started_at
        summary = summarize_yb171_v4_result(
            result=result,
            gate_time_ns=GATE_TIME_NS,
            omega_max_hz=OMEGA_MAX_HZ,
            model=model,
        )
        summary["stage"] = stage.label
        summary["stage_spec"] = asdict(stage)
        summary["wall_clock_stage_s"] = wall_clock
        stage_summaries.append(summary)
        if float(summary["probe_fidelity"]) > float(best_summary["probe_fidelity"]):
            best_summary = summary
            best_result = result
        current_ctrl_x = result.ctrl_x
        current_ctrl_y = result.ctrl_y
        current_theta = float(result.optimized_theta)

    payload = {
        "omega_max_hz": OMEGA_MAX_HZ,
        "omega_max_mhz": OMEGA_MAX_HZ / 1e6,
        "gate_time_ns": GATE_TIME_NS,
        "ensemble_size": ENSEMBLE_SIZE,
        "stages": [asdict(stage) for stage in STAGES],
        "stage_summaries": stage_summaries,
        "best_stage": best_summary["stage"],
        "best_probe_fidelity": best_summary["probe_fidelity"],
        "best_fid_err": best_summary["fid_err"],
        "best_optimized_theta": best_summary["optimized_theta"],
        "threshold_0p99_reached": bool(float(best_summary["probe_fidelity"]) >= 0.99),
        "threshold_0p999_reached": bool(float(best_summary["probe_fidelity"]) >= 0.999),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best_result is not None:
        result_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    else:
        result_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
