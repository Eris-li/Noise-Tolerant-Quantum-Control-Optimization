from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.ideal_cz import IdealCZModel
from neutral_yb.optimization.phase_grape import PhaseOnlyGrapeConfig, PhaseOnlyGrapeOptimizer


def main() -> None:
    species = idealised_yb171()
    model = IdealCZModel(species=species)
    config = PhaseOnlyGrapeConfig(
        num_tslots=40,
        evo_time=4.0,
        max_iter=200,
        leakage_weight=5.0,
        seed=7,
        init_phase_spread=0.2,
    )
    optimizer = PhaseOnlyGrapeOptimizer(model=model, config=config)
    summary = optimizer.optimize()

    artifact_path = ROOT / "artifacts" / "reference_2202_00903_ideal_yb_cz.json"
    optimizer.save_summary(summary, artifact_path)

    print("Reference experiment: ideal 171Yb CZ under infinite blockade")
    print(f"Success: {summary.success}")
    print(f"Iterations: {summary.iterations}")
    print(f"Objective: {summary.objective:.8f}")
    print(f"Optimized gamma: {summary.gamma:.8f}")
    print(f"Fidelity to CZ: {summary.fidelity_to_cz:.8f}")
    print(f"Leakage: {summary.leakage:.8e}")
    print(f"Best entangling-family phase: {summary.best_entangling_phase:.8f}")
    print(f"Fidelity to entangling family: {summary.fidelity_to_entangling_family:.8f}")
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()
