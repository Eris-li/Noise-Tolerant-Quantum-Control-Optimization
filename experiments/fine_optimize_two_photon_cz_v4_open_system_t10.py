from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def evaluate_zero_baseline() -> dict[str, float | str | bool]:
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=100,
            evo_time=10.0,
            max_iter=0,
            num_restarts=1,
            init_pulse_type="ZERO",
            init_control_scale=0.0,
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
            show_progress=False,
        ),
    )
    ctrl_x = np.zeros(optimizer.config.num_tslots, dtype=np.float64)
    ctrl_y = np.zeros(optimizer.config.num_tslots, dtype=np.float64)
    final_state = optimizer.final_phase_state(ctrl_x, ctrl_y)
    theta, fidelity = optimizer.model.optimize_theta_for_ket(final_state)
    return {
        "stage_name": "zero_baseline",
        "probe_fidelity": float(fidelity),
        "fid_err": float(1.0 - fidelity),
        "optimized_theta": float(theta),
        "num_tslots": int(optimizer.config.num_tslots),
        "max_iter": 0,
        "num_restarts": 1,
        "success": True,
    }


def run_stage(
    *,
    stage_name: str,
    initial_ctrl_x,
    initial_ctrl_y,
    initial_theta,
    num_tslots: int,
    max_iter: int,
    num_restarts: int,
    init_pulse_type: str,
    init_control_scale: float,
    control_smoothness_weight: float,
    control_curvature_weight: float,
) -> dict[str, object]:
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=num_tslots,
            evo_time=10.0,
            max_iter=max_iter,
            num_restarts=num_restarts,
            seed=17,
            init_pulse_type=init_pulse_type,
            init_control_scale=init_control_scale,
            control_smoothness_weight=control_smoothness_weight,
            control_curvature_weight=control_curvature_weight,
            fidelity_target=0.999,
            show_progress=True,
        ),
    )
    result = optimizer.optimize(
        initial_ctrl_x=initial_ctrl_x,
        initial_ctrl_y=initial_ctrl_y,
        initial_theta=initial_theta,
    )

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_result(result, artifacts / f"two_photon_cz_v4_open_system_t10_{stage_name}.json")

    return {
        "stage_name": stage_name,
        "config": {
            "num_tslots": num_tslots,
            "max_iter": max_iter,
            "num_restarts": num_restarts,
        },
        "result": result,
    }


def main() -> None:
    stages = [
        {
            "stage_name": "sine_large_search",
            "num_tslots": 100,
            "max_iter": 80,
            "num_restarts": 8,
            "init_pulse_type": "SINE",
            "init_control_scale": 0.75,
            "control_smoothness_weight": 0.0,
            "control_curvature_weight": 0.0,
            "warm_start": False,
        },
        {
            "stage_name": "random_large_search",
            "num_tslots": 100,
            "max_iter": 80,
            "num_restarts": 8,
            "init_pulse_type": "RANDOM",
            "init_control_scale": 0.75,
            "control_smoothness_weight": 0.0,
            "control_curvature_weight": 0.0,
            "warm_start": False,
        },
        {
            "stage_name": "refine_best",
            "num_tslots": 100,
            "max_iter": 150,
            "num_restarts": 3,
            "init_pulse_type": "SINE",
            "init_control_scale": 0.75,
            "control_smoothness_weight": 0.0,
            "control_curvature_weight": 0.0,
            "warm_start": True,
        },
    ]

    stage_records: list[dict[str, object]] = []
    best_result = None
    best_ctrl_x = None
    best_ctrl_y = None
    best_theta = None

    artifacts = ROOT / "artifacts"
    interrupted = False

    try:
        baseline = evaluate_zero_baseline()
        stage_records.append(baseline)
        print(
            f"[t10-fine] baseline zero-control F={baseline['probe_fidelity']:.6f} "
            f"theta={baseline['optimized_theta']:.6f}",
            flush=True,
        )

        for stage in stages:
            stage_name = stage["stage_name"]
            num_tslots = stage["num_tslots"]
            max_iter = stage["max_iter"]
            num_restarts = stage["num_restarts"]
            init_pulse_type = stage["init_pulse_type"]
            init_control_scale = stage["init_control_scale"]
            control_smoothness_weight = stage["control_smoothness_weight"]
            control_curvature_weight = stage["control_curvature_weight"]
            warm_start = bool(stage["warm_start"])
            print(
                f"[t10-fine] stage={stage_name} slots={num_tslots} "
                f"max_iter={max_iter} restarts={num_restarts} "
                f"init={init_pulse_type} scale={init_control_scale}",
                flush=True,
            )
            record = run_stage(
                stage_name=stage_name,
                initial_ctrl_x=best_ctrl_x if warm_start else None,
                initial_ctrl_y=best_ctrl_y if warm_start else None,
                initial_theta=best_theta if warm_start else (np.pi / 2.0),
                num_tslots=num_tslots,
                max_iter=max_iter,
                num_restarts=num_restarts,
                init_pulse_type=init_pulse_type,
                init_control_scale=init_control_scale,
                control_smoothness_weight=control_smoothness_weight,
                control_curvature_weight=control_curvature_weight,
            )
            result = record["result"]
            stage_records.append(
                {
                    "stage_name": stage_name,
                    "num_tslots": num_tslots,
                    "max_iter": max_iter,
                    "num_restarts": num_restarts,
                    "init_pulse_type": init_pulse_type,
                    "init_control_scale": init_control_scale,
                    "control_smoothness_weight": control_smoothness_weight,
                    "control_curvature_weight": control_curvature_weight,
                    "warm_start": warm_start,
                    "probe_fidelity": result.probe_fidelity,
                    "fid_err": result.fid_err,
                    "wall_time": result.wall_time,
                    "num_iter": result.num_iter,
                    "num_fid_func_calls": result.num_fid_func_calls,
                    "optimized_theta": result.optimized_theta,
                }
            )
            if best_result is None or result.probe_fidelity > best_result.probe_fidelity:
                best_result = result
                best_ctrl_x, best_ctrl_y, best_theta = result.ctrl_x, result.ctrl_y, result.optimized_theta

            print(
                f"[t10-fine] completed stage={stage_name} "
                f"F={result.probe_fidelity:.6f} fid_err={result.fid_err:.6e}",
                flush=True,
            )
            if result.probe_fidelity >= 0.999:
                print(f"[t10-fine] threshold reached at stage={stage_name}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[t10-fine] interrupted; saving partial summary", flush=True)

    summary = {
        "target_evo_time": 10.0,
        "threshold": 0.999,
        "stages": stage_records,
        "best_probe_fidelity": None if best_result is None else best_result.probe_fidelity,
        "best_fid_err": None if best_result is None else best_result.fid_err,
        "best_optimized_theta": None if best_result is None else best_result.optimized_theta,
        "threshold_reached": False if best_result is None else best_result.probe_fidelity >= 0.999,
        "interrupted": interrupted,
    }
    (artifacts / "two_photon_cz_v4_open_system_t10_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if best_result is not None:
        OpenSystemGRAPEOptimizer(
            model=build_yb171_v4_calibrated_model(),
            config=OpenSystemGRAPEConfig(num_tslots=100, evo_time=10.0),
        ).save_result(best_result, artifacts / "two_photon_cz_v4_open_system_t10_best.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
