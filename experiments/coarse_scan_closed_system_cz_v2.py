from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.finite_blockade_cz_5d import FiniteBlockadeCZ5DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def main() -> None:
    model = FiniteBlockadeCZ5DModel(
        species=idealised_yb171(),
        blockade_shift=8.0,
        static_detuning_01=0.015,
        static_detuning_11=0.015,
        rabi_scale=0.985,
    )

    durations = list(reversed(frange(1.0, 10.0, 0.5)))
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=100,
            evo_time=durations[0],
            max_iter=250,
            phase_seed=11,
            init_phase_spread=0.8,
            fidelity_target=0.999,
            smoothness_weight=0.002,
            num_restarts=4,
        ),
    )
    scan, results = optimizer.scan_durations(durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_scan(scan, artifacts / "closed_system_cz_v2_coarse_scan.json")

    best = max(results, key=lambda item: item.fidelity)
    optimizer.save_result(best, artifacts / "closed_system_cz_v2_best.json")

    print("Closed-system corrected CZ coarse scan completed")
    print(f"Best T*Omega_max: {best.evo_time}")
    print(f"Best fidelity: {best.fidelity}")
    print("Saved coarse scan and best pulse under artifacts/")


if __name__ == "__main__":
    main()
