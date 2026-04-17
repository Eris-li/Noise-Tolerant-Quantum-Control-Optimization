from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def run_stage(
    *,
    stage_name: str,
    initial_ctrl_x,
    initial_ctrl_y,
    initial_theta,
    num_tslots: int,
    max_iter: int,
    num_restarts: int,
) -> dict[str, object]:
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=num_tslots,
            evo_time=10.0,
            max_iter=max_iter,
            num_restarts=num_restarts,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
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
        ("warmup", 100, 1, 1),
        ("medium", 100, 3, 3),
        ("heavy", 100, 5, 10),
    ]

    stage_records: list[dict[str, object]] = []
    best_result = None
    ctrl_x = None
    ctrl_y = None
    theta = None

    artifacts = ROOT / "artifacts"
    interrupted = False

    try:
        for stage_name, num_tslots, max_iter, num_restarts in stages:
            print(
                f"[t10-fine] stage={stage_name} slots={num_tslots} "
                f"max_iter={max_iter} restarts={num_restarts}",
                flush=True,
            )
            record = run_stage(
                stage_name=stage_name,
                initial_ctrl_x=ctrl_x,
                initial_ctrl_y=ctrl_y,
                initial_theta=theta,
                num_tslots=num_tslots,
                max_iter=max_iter,
                num_restarts=num_restarts,
            )
            result = record["result"]
            stage_records.append(
                {
                    "stage_name": stage_name,
                    "num_tslots": num_tslots,
                    "max_iter": max_iter,
                    "num_restarts": num_restarts,
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
            ctrl_x, ctrl_y, theta = result.ctrl_x, result.ctrl_y, result.optimized_theta

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
