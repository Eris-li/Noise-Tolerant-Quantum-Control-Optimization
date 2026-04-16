from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.spline_phase_grape import (
    SplinePhaseOptimizationConfig,
    SplinePhaseOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def build_model() -> TwoPhotonCZ9DModel:
    return TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
    )


def main() -> None:
    threshold = 0.9999
    durations = list(reversed(frange(1.0, 10.0, 0.5)))
    optimizer = SplinePhaseOptimizer(
        model=build_model(),
        config=SplinePhaseOptimizationConfig(
            num_tslots=120,
            num_nodes=16,
            evo_time=durations[0],
            max_iter=250,
            phase_seed=31,
            init_phase_spread=0.4,
            fidelity_target=threshold,
            smoothness_weight=0.01,
            curvature_weight=0.02,
            node_curvature_weight=0.01,
            num_restarts=4,
            show_progress=True,
        ),
    )
    scan, results = optimizer.scan_durations(durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_scan(scan, artifacts / "two_photon_cz_v4_spline_coarse_scan.json")

    qualifying = [res for res in results if res.fidelity >= threshold]
    if qualifying:
        best = min(qualifying, key=lambda res: res.evo_time)
    else:
        best = max(results, key=lambda res: res.fidelity)
    optimizer.save_result(best, artifacts / "two_photon_cz_v4_spline_best.json")

    print("Two-photon CZ v4 spline coarse scan completed")
    print(f"Best scanned point = {best.evo_time}")
    print(f"Best fidelity = {best.fidelity}")
    print(f"Threshold reached = {scan.target_reached}")


if __name__ == "__main__":
    main()
