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
from neutral_yb.models.finite_blockade_cz_5d import FiniteBlockadeCZ5DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)


def main() -> None:
    artifacts = ROOT / "artifacts"
    scan = json.loads((artifacts / "closed_system_cz_v2_coarse_scan.json").read_text(encoding="utf-8"))
    best = json.loads((artifacts / "closed_system_cz_v2_best.json").read_text(encoding="utf-8"))

    phases = np.asarray(best["phases"], dtype=np.float64)
    model = FiniteBlockadeCZ5DModel(
        species=idealised_yb171(),
        blockade_shift=8.0,
        static_detuning_01=0.015,
        static_detuning_11=0.015,
        rabi_scale=0.985,
    )
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=len(phases),
            evo_time=float(best["evo_time"]),
            max_iter=250,
        ),
    )
    times, states = optimizer.trajectory(phases)
    array = np.stack(states, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(scan["durations"], scan["fidelities"], marker="o", color="#1f78b4")
    ax.axhline(0.999, color="#d95f0e", linestyle="--", label="target fidelity 0.999")
    ax.axvline(best["evo_time"], color="#6a3d9a", linestyle=":", label=f"best {best['evo_time']:.2f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.5, 1.001)
    ax.set_title("Corrected closed-system coarse scan")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(phases)), centered_phase(phases), color="#6a3d9a")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Unwrapped phase [rad]")
    ax.set_title("Best phase sequence")

    ax = axes[1, 0]
    ax.plot(times, np.abs(array[:, 0]) ** 2, label="|01>", color="#1f78b4")
    ax.plot(times, np.abs(array[:, 1]) ** 2, label="|0r>", color="#33a02c")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Single-excitation branch")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, np.abs(array[:, 2]) ** 2, label="|11>", color="#e31a1c")
    ax.plot(times, np.abs(array[:, 3]) ** 2, label="|W>", color="#ff7f00")
    ax.plot(times, np.abs(array[:, 4]) ** 2, label="|rr>", color="#6a3d9a")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Double-excitation branch")
    ax.legend()

    fig.suptitle("Closed-system corrected CZ (finite blockade)")
    fig.tight_layout()

    output_path = artifacts / "closed_system_cz_v2_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()

