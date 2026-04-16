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

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.optimization.spline_phase_grape import (
    SplinePhaseOptimizationConfig,
    SplinePhaseOptimizer,
)


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)


def main() -> None:
    artifacts = ROOT / "artifacts"
    scan = json.loads((artifacts / "two_photon_cz_v4_spline_coarse_scan.json").read_text(encoding="utf-8"))
    optimal = json.loads((artifacts / "two_photon_cz_v4_spline_best.json").read_text(encoding="utf-8"))

    phases = np.asarray(optimal["phases"], dtype=np.float64)
    node_phases = np.asarray(optimal["node_phases"], dtype=np.float64)

    model = TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
    )
    optimizer = SplinePhaseOptimizer(
        model=model,
        config=SplinePhaseOptimizationConfig(
            num_tslots=len(phases),
            num_nodes=len(node_phases),
            evo_time=float(optimal["evo_time"]),
            max_iter=250,
            fidelity_target=0.9999,
            smoothness_weight=0.01,
            curvature_weight=0.02,
            node_curvature_weight=0.01,
            num_restarts=4,
        ),
    )
    times, states = optimizer.trajectory(node_phases)
    array = np.stack(states, axis=0)
    phase_jumps = np.diff(np.unwrap(phases))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(scan["durations"], scan["fidelities"], marker="o", linewidth=2.0, markersize=5, color="#1f78b4")
    ax.axhline(0.9999, color="#d95f0e", linestyle="--", label="1 - F = 1e-4")
    ax.axvline(optimal["evo_time"], color="#6a3d9a", linestyle=":", label=f"T={optimal['evo_time']:.3f}")
    ax.set_xlabel("T * Omega_eff")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.35, 1.001)
    ax.set_title("Two-photon CZ spline coarse scan")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(phases)), centered_phase(phases), color="#1f78b4", label="slice phases")
    node_positions = np.linspace(0, len(phases) - 1, len(node_phases))
    ax.scatter(node_positions, centered_phase(node_phases), color="#e31a1c", label="spline nodes", zorder=3)
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Phase [rad]")
    ax.set_title("Spline phase profile")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.arange(len(phase_jumps)), phase_jumps, color="#1f78b4", label="adjacent phase jump")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Adjacent phase change [rad]")
    ax.set_title("Phase jumps from spline-expanded profile")
    ax.legend()

    ax = axes[1, 1]
    intermediate_total = np.abs(array[:, 1]) ** 2 + np.abs(array[:, 4]) ** 2 + np.abs(array[:, 5]) ** 2
    rydberg_total = np.abs(array[:, 2]) ** 2 + np.abs(array[:, 6]) ** 2 + np.abs(array[:, 7]) ** 2 + np.abs(array[:, 8]) ** 2
    ax.plot(times, np.abs(array[:, 0]) ** 2, label="|01>", color="#1f78b4")
    ax.plot(times, np.abs(array[:, 3]) ** 2, label="|11>", color="#e31a1c")
    ax.plot(times, intermediate_total, label="intermediate total", color="#33a02c")
    ax.plot(times, rydberg_total, label="rydberg total", color="#6a3d9a")
    ax.set_xlabel("t Omega_eff")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population dynamics at spline optimum")
    ax.legend()

    fig.suptitle("Two-photon CZ with spline-node single-phase control")
    fig.tight_layout()

    output_path = artifacts / "two_photon_cz_v4_spline_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
