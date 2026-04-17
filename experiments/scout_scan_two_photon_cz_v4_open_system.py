from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.optimization.open_system_grape import (
    OpenSystemGRAPEConfig,
    OpenSystemGRAPEOptimizer,
)


def main() -> None:
    durations = [10.0, 9.0, 8.0, 7.5]
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=durations[0],
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
    scan, results = optimizer.scan_durations(durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_scan(scan, artifacts / "two_photon_cz_v4_open_system_scout.json")

    summary = {
        "durations": scan.durations,
        "fidelities": scan.fidelities,
        "best_fidelity": scan.best_fidelity,
        "best_duration": scan.best_duration,
        "target_reached": scan.target_reached,
        "num_tslots": optimizer.config.num_tslots,
        "max_iter": optimizer.config.max_iter,
        "num_restarts": optimizer.config.num_restarts,
        "points": [
            {
                "evo_time": result.evo_time,
                "probe_fidelity": result.probe_fidelity,
                "fid_err": result.fid_err,
                "wall_time": result.wall_time,
                "optimized_theta": result.optimized_theta,
            }
            for result in results
        ],
    }
    destination = artifacts / "two_photon_cz_v4_open_system_scout_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Scout scan completed")
    print(f"Saved scan to {artifacts / 'two_photon_cz_v4_open_system_scout.json'}")
    print(f"Saved summary to {destination}")


if __name__ == "__main__":
    main()
