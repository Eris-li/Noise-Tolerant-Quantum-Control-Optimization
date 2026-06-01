from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from neutral_yb.config.artifact_paths import evered2023_parallel_cz_dir
from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.evered2023_parallel_cz import (
    Evered2023TimeOptimalPulse,
    build_evered2023_two_photon_detuning_model,
    build_evered2023_two_photon_ladder_model,
)
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.grape import ClosedSystemGRAPE
from neutral_yb.optimization.evered2023_parameterized_grape import (
    Evered2023ParameterizedGRAPEConfig,
    Evered2023ParameterizedGRAPEResult,
)


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(phases, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)


def main() -> None:
    artifacts = evered2023_parallel_cz_dir(ROOT)
    summary = json.loads((artifacts / "evered2023_parameterized_grape_scan.json").read_text(encoding="utf-8"))
    scan = summary["scan"]
    results = scan["results"]
    target = float(scan["target_fidelity"])
    threshold_result = scan["best_threshold_result"]
    best = scan["best_fidelity_result"]
    paper = Evered2023TimeOptimalPulse()

    durations = np.asarray(scan["durations_omega_t_over_2pi"], dtype=np.float64)
    fidelities = np.asarray(scan["fidelities"], dtype=np.float64)

    optimized_pulse = Evered2023TimeOptimalPulse(
        amplitude_phase_modulation=float(best["amplitude_phase_modulation_rad"]),
        phase_rate=float(best["phase_rate_over_omega"]),
        phase_offset=float(best["phase_offset_rad"]),
        static_detuning=float(best["static_detuning_over_omega"]),
        omega_t_over_2pi=float(best["omega_t_over_2pi"]),
    )

    num_tslots = int(best["num_tslots"])
    times = (np.arange(num_tslots, dtype=np.float64) + 0.5) * (
        optimized_pulse.dimensionless_duration / num_tslots
    )
    optimized_phases = optimized_pulse.phase(times)
    paper_reference = Evered2023TimeOptimalPulse(omega_t_over_2pi=optimized_pulse.omega_t_over_2pi)
    paper_phases = paper_reference.phase(times)

    setup = summary.get("grape_setup", {})
    if setup.get("model") == "two-photon-detuning":
        h_params = summary["hamiltonian_parameters"]
        model = build_evered2023_two_photon_detuning_model(
            species=idealised_yb171(),
            intermediate_detuning_over_effective_rabi=float(h_params["intermediate_detuning"]),
            blockade_shift_over_effective_rabi=float(h_params["blockade_shift"]),
        )
        trajectory_optimizer = ClosedSystemGRAPE.evered_detuning(
            model=model,
            omega_t_over_2pi=optimized_pulse.omega_t_over_2pi,
            config=Evered2023ParameterizedGRAPEConfig(num_tslots=num_tslots, num_restarts=1),
        )
        reconstructed_result = Evered2023ParameterizedGRAPEResult(
            omega_t_over_2pi=float(best["omega_t_over_2pi"]),
            amplitude_phase_modulation=float(best["amplitude_phase_modulation_rad"]),
            phase_rate=float(best["phase_rate_over_omega"]),
            phase_offset=float(best["phase_offset_rad"]),
            static_detuning=float(best["static_detuning_over_omega"]),
            theta=float(best["theta"]),
            fidelity=float(best["fidelity"]),
            objective=float(best["objective"]),
            iterations=int(best["iterations"]),
            success=bool(best["success"]),
            message=str(best["message"]),
            num_tslots=int(best["num_tslots"]),
            num_restarts=int(best["num_restarts"]),
            wall_time=float(best["wall_time"]),
        )
        traj_times, states = trajectory_optimizer.trajectory(reconstructed_result)
    elif setup.get("model") == "two-photon":
        h_params = summary["hamiltonian_parameters"]
        model = build_evered2023_two_photon_ladder_model(
            species=idealised_yb171(),
            lower_rabi=float(h_params["blue_rabi"]),
            upper_rabi=float(h_params["red_rabi"]),
            intermediate_detuning=float(h_params["intermediate_detuning"]),
            blockade_shift=float(h_params["blockade_shift"]),
        )
        optimizer = ClosedSystemGRAPE.evered_parameterized(
            model=model,
            omega_t_over_2pi=optimized_pulse.omega_t_over_2pi,
            config=Evered2023ParameterizedGRAPEConfig(num_tslots=num_tslots, num_restarts=1),
        )
        trajectory_optimizer = optimizer.slot_optimizer
        traj_times, states = trajectory_optimizer.trajectory(optimized_phases)
    else:
        model = GlobalCZ4DModel(species=idealised_yb171())
        optimizer = ClosedSystemGRAPE.evered_parameterized(
            model=model,
            omega_t_over_2pi=optimized_pulse.omega_t_over_2pi,
            config=Evered2023ParameterizedGRAPEConfig(num_tslots=num_tslots, num_restarts=1),
        )
        trajectory_optimizer = optimizer.slot_optimizer
        traj_times, states = trajectory_optimizer.trajectory(optimized_phases)
    state_array = np.stack(states, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(durations, fidelities, marker="o", linewidth=2.0, markersize=4, color="#1f78b4", label="parameterized GRAPE")
    ax.axhline(target, color="#d95f0e", linestyle="--", label=f"target F={target:.5f}")
    ax.axvline(paper.omega_t_over_2pi, color="#000000", linestyle="-.", label="paper OmegaT/2pi=1.215")
    if threshold_result is not None:
        ax.axvline(
            float(threshold_result["omega_t_over_2pi"]),
            color="#6a3d9a",
            linestyle=":",
            label=f"first target {float(threshold_result['omega_t_over_2pi']):.3f}",
        )
    ax.axvline(
        float(best["omega_t_over_2pi"]),
        color="#4b0082",
        linestyle="-",
        alpha=0.45,
        label=f"best {float(best['omega_t_over_2pi']):.3f}",
    )
    ax.set_xlabel("Omega T / 2pi")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(max(0.98, float(np.min(fidelities)) - 0.002), 1.00005)
    ax.set_title("Time Scan")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(times / (2.0 * np.pi), centered_phase(optimized_phases), color="#6a3d9a", linewidth=2.0, label="GRAPE fit")
    ax.plot(times / (2.0 * np.pi), centered_phase(paper_phases), color="#33a02c", linestyle="--", linewidth=2.0, label="paper Eq. (1)")
    ax.set_xlabel("Omega t / 2pi")
    ax.set_ylabel("Unwrapped phase [rad]")
    ax.set_title("Recovered Phase Profile")
    ax.legend()

    ax = axes[1, 0]
    labels = ["A/2pi", "omega/Omega", "phi0", "delta0/Omega"]
    optimized_values = [
        float(best["amplitude_phase_modulation_over_2pi"]),
        float(best["phase_rate_over_omega"]),
        float(best["phase_offset_rad"]),
        float(best["static_detuning_over_omega"]),
    ]
    paper_values = [
        paper.amplitude_phase_modulation / (2.0 * np.pi),
        paper.phase_rate,
        paper.phase_offset,
        paper.static_detuning,
    ]
    colors = ["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a"]
    differences = np.asarray(optimized_values, dtype=np.float64) - np.asarray(paper_values, dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    ax.axhline(0.0, color="#000000", linestyle="--", linewidth=1.0)
    ax.bar(x, differences, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GRAPE - paper")
    ax.set_title("Parameter Difference")

    ax = axes[1, 1]
    if setup.get("model") in {"two-photon", "two-photon-detuning"}:
        labels = model.basis_labels()
        colors = ["#1f78b4", "#8dd3c7", "#33a02c", "#e31a1c", "#fb9a99", "#b15928", "#ff7f00", "#cab2d6", "#6a3d9a"]
        for index, label in enumerate(labels):
            population = np.abs(state_array[:, index]) ** 2
            if float(np.max(population)) > 1e-3 or index in (0, 3):
                ax.plot(traj_times / (2.0 * np.pi), population, label=f"|{label}>", color=colors[index])
    else:
        ax.plot(traj_times / (2.0 * np.pi), np.abs(state_array[:, 0]) ** 2, label="|01>", color="#1f78b4")
        ax.plot(traj_times / (2.0 * np.pi), np.abs(state_array[:, 1]) ** 2, label="|0r>", color="#33a02c")
        ax.plot(traj_times / (2.0 * np.pi), np.abs(state_array[:, 2]) ** 2, label="|11>", color="#e31a1c")
        ax.plot(traj_times / (2.0 * np.pi), np.abs(state_array[:, 3]) ** 2, label="|W>", color="#ff7f00")
    ax.set_xlabel("Omega t / 2pi")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population Dynamics")
    ax.legend()

    fig.suptitle("Evered 2023 fixed-amplitude CZ: GRAPE validation")
    fig.tight_layout()

    output = artifacts / "evered2023_parameterized_grape_summary.png"
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
