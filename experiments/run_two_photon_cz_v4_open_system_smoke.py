from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def main() -> None:
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=8.5,
            max_iter=8,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            show_progress=True,
        ),
    )
    result = optimizer.optimize()

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "two_photon_cz_v4_open_system_smoke.json"
    optimizer.save_result(result, destination)

    print("Two-photon CZ v4 open-system smoke run completed")
    print(f"Probe fidelity = {result.probe_fidelity}")
    print(f"Objective = {result.objective}")
    print(f"Fid err = {result.fid_err}")
    print(f"Optimized theta = {result.optimized_theta}")
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
