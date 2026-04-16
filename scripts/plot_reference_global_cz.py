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
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def main() -> None:
    artifact_path = ROOT / "artifacts" / "reference_2202_00903_global_cz.json"
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    phases = np.asarray(summary["phases"], dtype=np.float64)

    model = GlobalCZ4DModel(species=idealised_yb171())
    config = GlobalPhaseOptimizationConfig(
        num_tslots=len(phases),
        evo_time=float(summary["evo_time"]),
        max_iter=300,
        phase_seed=11,
        init_phase_spread=0.8,
    )
    optimizer = PaperGlobalPhaseOptimizer(model=model, config=config)
    times, states = optimizer.trajectory(phases)
    array = np.stack(states, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(["Fidelity", "1 - Fidelity"], [summary["fidelity"], 1.0 - summary["fidelity"]], color=["#2c7fb8", "#d95f0e"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase-Gate Fidelity")

    ax = axes[0, 1]
    ax.step(np.arange(len(phases)), phases, where="mid", color="#6a3d9a")
    ax.set_title("Optimized Laser Phase")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Phase [rad]")
    ax.set_ylim(-0.1, 2.0 * np.pi + 0.1)

    ax = axes[1, 0]
    ax.plot(times, np.abs(array[:, 0]) ** 2, label="|01>", color="#1f78b4")
    ax.plot(times, np.abs(array[:, 1]) ** 2, label="|0r>", color="#33a02c")
    ax.set_title("Block H01")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, np.abs(array[:, 2]) ** 2, label="|11>", color="#e31a1c")
    ax.plot(times, np.abs(array[:, 3]) ** 2, label="|W>", color="#ff7f00")
    ax.set_title("Block H11")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    fig.suptitle("Reference Experiment: 2202.00903-style Global CZ")
    fig.tight_layout()

    output_path = ROOT / "artifacts" / "reference_2202_00903_global_cz.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()

