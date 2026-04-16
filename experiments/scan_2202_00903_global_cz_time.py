from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def main() -> None:
    durations = [7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8]
    model = GlobalCZ4DModel(species=idealised_yb171())
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=100,
            evo_time=durations[-1],
            max_iter=250,
            phase_seed=11,
            init_phase_spread=0.8,
            fidelity_target=0.999,
        ),
    )
    scan, _ = optimizer.scan_durations(durations)

    artifact_path = ROOT / "artifacts" / "reference_2202_00903_global_cz_scan.json"
    optimizer.save_scan(scan, artifact_path)

    print("Time scan for paper-style global CZ")
    print(f"Best fidelity: {scan.best_fidelity:.10f}")
    print(f"Target reached: {scan.target_reached}")
    print(f"Best duration: {scan.best_duration}")
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()

