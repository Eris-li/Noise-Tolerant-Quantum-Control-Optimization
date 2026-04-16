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
    model = GlobalCZ4DModel(species=idealised_yb171())
    config = GlobalPhaseOptimizationConfig(
        num_tslots=100,
        evo_time=8.8,
        max_iter=300,
        phase_seed=11,
        init_phase_spread=0.8,
        fidelity_target=0.999,
    )
    optimizer = PaperGlobalPhaseOptimizer(model=model, config=config)
    result = optimizer.optimize()

    artifact_path = ROOT / "artifacts" / "reference_2202_00903_global_cz.json"
    optimizer.save_result(result, artifact_path)

    print("Reference experiment: paper-style global CZ in 4D symmetry-reduced space")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Objective: {result.objective:.10f}")
    print(f"Fidelity: {result.fidelity:.10f}")
    print(f"Theta: {result.theta:.10f}")
    print(f"T * Omega_max: {result.evo_time:.10f}")
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()

