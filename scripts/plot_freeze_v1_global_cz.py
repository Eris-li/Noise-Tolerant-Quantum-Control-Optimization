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

from neutral_yb.config.artifact_paths import v1_artifacts_dir
from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)


def main() -> None:
    artifacts = v1_artifacts_dir(ROOT)
    coarse = json.loads((artifacts / "freeze_v1_global_cz_coarse_scan.json").read_text(encoding="utf-8"))
    fine = json.loads((artifacts / "freeze_v1_global_cz_fine_scan.json").read_text(encoding="utf-8"))
    optimal = json.loads((artifacts / "freeze_v1_global_cz_optimal.json").read_text(encoding="utf-8"))
    fit = json.loads((artifacts / "freeze_v1_global_cz_fit.json").read_text(encoding="utf-8"))

    phases = np.asarray(optimal["phases"], dtype=np.float64)
    centered = centered_phase(phases)

    model = GlobalCZ4DModel(species=idealised_yb171())
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=len(phases),
            evo_time=float(optimal["evo_time"]),
            max_iter=300,
        ),
    )
    times, states = optimizer.trajectory(phases)
    array = np.stack(states, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(coarse["durations"], coarse["fidelities"], marker="o", linewidth=2.0, markersize=5, color="#1f78b4", label="coarse")
    ax.axhline(0.9999, color="#d95f0e", linestyle="--", label="1 - F = 1e-4")
    ax.axvline(optimal["evo_time"], color="#6a3d9a", linestyle=":", label=f"threshold point {optimal['evo_time']:.3f}")
    ax.axvline(fit["t_star"], color="#000000", linestyle="-.", label=f"fit T*={fit['t_star']:.3f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.58, 1.001)
    ax.set_title("Coarse Scan (global view)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(fine["durations"], fine["fidelities"], marker=".", linewidth=1.5, markersize=6, color="#33a02c", label="fine")
    ax.axhline(0.9999, color="#d95f0e", linestyle="--", label="1 - F = 1e-4")
    ax.axvline(optimal["evo_time"], color="#6a3d9a", linestyle=":", label=f"T*={optimal['evo_time']:.3f}")
    ax.axvline(fit["t_star"], color="#000000", linestyle="-.", label=f"fit T*={fit['t_star']:.3f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.9992, 1.00002)
    ax.set_title("Fine Scan (threshold zoom)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.arange(len(phases)), centered, color="#6a3d9a")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Unwrapped phase [rad]")
    ax.set_title("Candidate time-optimal phase sequence")

    ax = axes[1, 1]
    ax.plot(times, np.abs(array[:, 0]) ** 2, label="|01>", color="#1f78b4")
    ax.plot(times, np.abs(array[:, 1]) ** 2, label="|0r>", color="#33a02c")
    ax.plot(times, np.abs(array[:, 2]) ** 2, label="|11>", color="#e31a1c")
    ax.plot(times, np.abs(array[:, 3]) ** 2, label="|W>", color="#ff7f00")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population dynamics at candidate optimum")
    ax.legend()

    fig.suptitle("Freeze candidate for 2202.00903-style global CZ")
    fig.tight_layout()

    output_path = artifacts / "freeze_v1_global_cz_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
