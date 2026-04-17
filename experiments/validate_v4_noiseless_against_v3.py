from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy.linalg import expm
import qutip

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.amplitude_phase_grape import (
    AmplitudePhaseOptimizationConfig,
    AmplitudePhaseOptimizer,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def build_v3_model() -> TwoPhotonCZ9DModel:
    return TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
    )


def build_v4_noiseless_model() -> TwoPhotonCZOpen10DModel:
    return TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
        noise=TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.0,
            common_two_photon_detuning=0.0,
            differential_two_photon_detuning=0.0,
            doppler_detuning_01=0.0,
            doppler_detuning_11=0.0,
            blockade_shift_offset=0.0,
            lower_amplitude_scale=1.0,
            upper_amplitude_scale=1.0,
            intermediate_decay_rate=0.0,
            rydberg_decay_rate=0.0,
            intermediate_dephasing_rate=0.0,
            rydberg_dephasing_rate=0.0,
            extra_rydberg_leakage_rate=0.0,
            intermediate_branch_to_qubit=1.0,
            rydberg_branch_to_qubit=1.0,
        ),
    )


def propagate_v3_probe(
    model: TwoPhotonCZ9DModel,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    probe: qutip.Qobj,
    evo_time: float,
) -> qutip.Qobj:
    h_d = np.asarray(model.drift_hamiltonian().full(), dtype=np.complex128)
    h_x, h_y = [np.asarray(operator.full(), dtype=np.complex128) for operator in model.phase_control_hamiltonians()[0]]
    state = np.asarray(probe.full(), dtype=np.complex128).ravel()
    dt = evo_time / amplitudes.size
    for amplitude, phase in zip(amplitudes, phases):
        h_k = h_d + float(amplitude) * (np.cos(float(phase)) * h_x + np.sin(float(phase)) * h_y)
        state = expm(-1j * dt * h_k) @ state
    return qutip.Qobj(state.reshape((-1, 1)))


def embed_probe_into_v3(active_probe: qutip.Qobj) -> qutip.Qobj:
    vector = np.zeros((9, 1), dtype=np.complex128)
    active = np.asarray(active_probe.full(), dtype=np.complex128).ravel()
    vector[0, 0] = active[0]
    vector[3, 0] = active[3]
    return qutip.Qobj(vector)


def main() -> None:
    evo_time = 8.0
    num_tslots = 100

    v3_model = build_v3_model()
    v4_model = build_v4_noiseless_model()

    v3_optimizer = AmplitudePhaseOptimizer(
        v3_model,
        AmplitudePhaseOptimizationConfig(
            num_tslots=num_tslots,
            evo_time=evo_time,
            max_iter=1,
            seed=17,
        ),
    )
    amplitudes, phases = v3_optimizer.initial_guess()
    ctrl_x = amplitudes * np.cos(phases)
    ctrl_y = amplitudes * np.sin(phases)

    v4_optimizer = OpenSystemGRAPEOptimizer(
        v4_model,
        OpenSystemGRAPEConfig(
            num_tslots=num_tslots,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
    )

    drift_error = float(
        np.max(
            np.abs(
                np.asarray(v3_model.drift_hamiltonian().full(), dtype=np.complex128)
                - np.asarray(v4_model.drift_hamiltonian().full(), dtype=np.complex128)[:9, :9]
            )
        )
    )
    v3_hx, v3_hy = [np.asarray(operator.full(), dtype=np.complex128) for operator in v3_model.phase_control_hamiltonians()[0]]
    v4_hx, v4_hy = [np.asarray(operator.full(), dtype=np.complex128)[:9, :9] for operator in v4_model.lower_leg_control_hamiltonians()]
    lower_x_error = float(np.max(np.abs(v3_hx - v4_hx)))
    lower_y_error = float(np.max(np.abs(v3_hy - v4_hy)))

    v3_final_states: list[qutip.Qobj] = []
    v4_final_states: list[qutip.Qobj] = []
    density_differences: list[float] = []
    loss_populations: list[float] = []
    purities: list[float] = []

    for active_probe, _ in v4_model.probe_kets(theta=0.0):
        v3_probe = embed_probe_into_v3(active_probe)
        v3_final = propagate_v3_probe(v3_model, amplitudes, phases, v3_probe, evo_time)
        v3_final_states.append(qutip.ket2dm(v3_final))

        v4_final = v4_optimizer.evolve_density_matrix(ctrl_x, ctrl_y, qutip.ket2dm(active_probe))
        v4_final_states.append(v4_final)

        v4_projected = np.asarray(v4_final.full(), dtype=np.complex128)[:9, :9]
        v3_density = np.asarray(qutip.ket2dm(v3_final).full(), dtype=np.complex128)
        density_differences.append(float(np.linalg.norm(v4_projected - v3_density)))
        loss_populations.append(float(np.real(v4_final.full()[9, 9])))
        purities.append(float((v4_final * v4_final).tr().real))

    v3_as_v4_densities = []
    for density in v3_final_states:
        embedded = np.zeros((10, 10), dtype=np.complex128)
        embedded[:9, :9] = np.asarray(density.full(), dtype=np.complex128)
        v3_as_v4_densities.append(qutip.Qobj(embedded))

    theta_v3, fidelity_v3 = v4_model.optimize_theta_for_probe_states(v3_as_v4_densities)
    theta_v4, fidelity_v4 = v4_model.optimize_theta_for_probe_states(v4_final_states)

    summary = {
        "evo_time": evo_time,
        "num_tslots": num_tslots,
        "drift_max_abs_error": drift_error,
        "lower_x_max_abs_error": lower_x_error,
        "lower_y_max_abs_error": lower_y_error,
        "max_density_frobenius_error": float(max(density_differences)),
        "max_loss_population": float(max(loss_populations)),
        "min_purity": float(min(purities)),
        "theta_v3_limit": float(theta_v3),
        "theta_v4_noiseless": float(theta_v4),
        "probe_fidelity_v3_limit": float(fidelity_v3),
        "probe_fidelity_v4_noiseless": float(fidelity_v4),
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "two_photon_cz_v4_noiseless_validation.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Validation written to {destination}")


if __name__ == "__main__":
    main()
