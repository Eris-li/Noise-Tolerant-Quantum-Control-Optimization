from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.config.artifact_paths import ensure_artifact_dir, v3_artifacts_dir
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.amplitude_phase_grape import (
    AmplitudePhaseOptimizationConfig,
    AmplitudePhaseOptimizer,
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
    durations = list(reversed(frange(7.5, 8.5, 0.2)))
    optimizer = AmplitudePhaseOptimizer(
        model=build_model(),
        config=AmplitudePhaseOptimizationConfig(
            num_tslots=100,
            evo_time=durations[0],
            max_iter=320,
            seed=17,
            init_phase_spread=0.35,
            init_amplitude_scale=0.75,
            fidelity_target=threshold,
            phase_smoothness_weight=0.01,
            phase_curvature_weight=0.02,
            amplitude_smoothness_weight=0.01,
            amplitude_curvature_weight=0.01,
            num_restarts=8,
            show_progress=True,
        ),
    )

    scan, results = optimizer.scan_durations(durations)

    artifacts = ensure_artifact_dir(v3_artifacts_dir(ROOT))
    optimizer.save_scan(scan, artifacts / "two_photon_cz_v3_local_scan_7p5_8p5.json")

    if results:
        best = max(results, key=lambda res: res.fidelity)
        optimizer.save_result(best, artifacts / "two_photon_cz_v3_local_scan_7p5_8p5_best.json")

    print("Two-photon CZ v3 local scan completed")
    print(f"Best fidelity in local scan = {scan.best_fidelity}")
    print(f"Earliest local threshold point = {scan.best_duration}")


if __name__ == "__main__":
    main()
