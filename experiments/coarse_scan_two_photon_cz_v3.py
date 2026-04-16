from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.linear_control_grape import (
    LinearControlGRAPEOptimizer,
    LinearControlOptimizationConfig,
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
    optimizer = LinearControlGRAPEOptimizer(
        model=build_model(),
        config=LinearControlOptimizationConfig(
            num_tslots=120,
            evo_time=durations[0],
            max_iter=220,
            control_seed=17,
            init_control_scale=0.1,
            control_bound=2.0,
            fidelity_target=threshold,
            smoothness_weight=0.01,
            amplitude_weight=0.0005,
            curvature_weight=0.02,
            num_restarts=4,
        ),
    )

    scan, results = optimizer.scan_durations(durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_scan(scan, artifacts / "two_photon_cz_v3_coarse_scan.json")

    qualifying = [res for res in results if res.fidelity >= threshold]
    if not qualifying:
        raise RuntimeError("No coarse-scan point reached fidelity >= 0.9999")
    threshold_candidate = min(qualifying, key=lambda res: res.evo_time)
    refined_optimizer = LinearControlGRAPEOptimizer(
        model=build_model(),
        config=LinearControlOptimizationConfig(
            num_tslots=threshold_candidate.num_tslots,
            evo_time=threshold_candidate.evo_time,
            max_iter=400,
            control_seed=17,
            init_control_scale=0.1,
            control_bound=2.0,
            fidelity_target=threshold,
            smoothness_weight=0.01,
            amplitude_weight=0.0005,
            curvature_weight=0.02,
            num_restarts=6,
        ),
    )
    optimal = refined_optimizer.optimize(
        initial_controls=threshold_candidate.controls,
        initial_theta=threshold_candidate.theta,
    )
    refined_optimizer.save_result(optimal, artifacts / "two_photon_cz_v3_best.json")

    print("Two-photon CZ v3 coarse scan completed")
    print(f"Earliest coarse threshold point = {optimal.evo_time}")
    print(f"Fidelity at threshold point = {optimal.fidelity}")
    print(f"Theta at threshold point = {optimal.theta}")


if __name__ == "__main__":
    main()
