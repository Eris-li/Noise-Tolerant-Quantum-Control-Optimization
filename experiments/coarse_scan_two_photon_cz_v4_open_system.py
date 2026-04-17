from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import (
    OpenSystemGRAPEConfig,
    OpenSystemGRAPEOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def build_model() -> TwoPhotonCZOpen10DModel:
    return TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
        noise=TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.01,
            common_two_photon_detuning=0.004,
            differential_two_photon_detuning=0.003,
            doppler_detuning_01=0.002,
            doppler_detuning_11=0.004,
            lower_amplitude_scale=0.99,
            upper_amplitude_scale=0.99,
            intermediate_decay_rate=0.025,
            rydberg_decay_rate=0.015,
            intermediate_dephasing_rate=0.004,
            rydberg_dephasing_rate=0.01,
            extra_rydberg_leakage_rate=0.003,
            intermediate_branch_to_qubit=0.45,
            rydberg_branch_to_qubit=0.05,
        ),
    )


def main() -> None:
    threshold = 0.995
    durations = list(reversed(frange(1.0, 10.0, 1.0)))
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=8,
            evo_time=durations[0],
            max_iter=5,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
            show_progress=True,
        ),
    )
    scan, results = optimizer.scan_durations(durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    optimizer.save_scan(scan, artifacts / "two_photon_cz_v4_open_system_coarse.json")

    best = max(results, key=lambda result: result.probe_fidelity)
    optimizer.save_result(best, artifacts / "two_photon_cz_v4_open_system_best.json")

    qualifying = [result for result in results if result.probe_fidelity >= threshold]
    if qualifying:
        optimal = min(qualifying, key=lambda result: result.evo_time)
        optimizer.save_result(optimal, artifacts / "two_photon_cz_v4_open_system_optimal.json")
        print(f"Earliest threshold point = {optimal.evo_time}")
        print(f"Threshold fidelity = {optimal.probe_fidelity}")
    else:
        print("No coarse-scan point reached the fidelity threshold")

    print("Two-photon CZ v4 open-system coarse scan completed")
    print(f"Best coarse T*Omega_max = {best.evo_time}")
    print(f"Best coarse probe fidelity = {best.probe_fidelity}")
    print(f"Best coarse theta = {best.optimized_theta}")


if __name__ == "__main__":
    main()
