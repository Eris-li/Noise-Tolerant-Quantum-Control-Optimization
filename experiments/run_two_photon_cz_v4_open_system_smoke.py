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
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


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
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=40,
            evo_time=8.5,
            max_iter=30,
            max_wall_time=900.0,
            fid_err_targ=5e-2,
            min_grad=1e-7,
            num_restarts=2,
            seed=17,
            init_pulse_type="SINE",
            pulse_scaling=0.3,
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
    print(f"Fid err = {result.fid_err}")
    print(f"Optimized theta = {result.optimized_theta}")
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
