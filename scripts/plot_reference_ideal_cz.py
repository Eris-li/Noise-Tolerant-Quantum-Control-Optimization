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
from neutral_yb.models.ideal_cz import IdealCZModel
from neutral_yb.optimization.phase_grape import PhaseOnlyGrapeConfig, PhaseOnlyGrapeOptimizer


def computational_basis_state(model: IdealCZModel, label: str) -> np.ndarray:
    mapping = {"00": 0, "01": 1, "10": 3, "11": 5}
    state = np.zeros(model.dimension(), dtype=np.complex128)
    state[mapping[label]] = 1.0
    return state


def state_populations(states: list[np.ndarray]) -> dict[str, np.ndarray]:
    array = np.stack(states, axis=0)
    return {
        "01": np.abs(array[:, 1]) ** 2,
        "0r": np.abs(array[:, 2]) ** 2,
        "10": np.abs(array[:, 3]) ** 2,
        "r0": np.abs(array[:, 4]) ** 2,
        "11": np.abs(array[:, 5]) ** 2,
        "W": np.abs(array[:, 6]) ** 2,
    }


def main() -> None:
    artifact_path = ROOT / "artifacts" / "reference_2202_00903_ideal_yb_cz.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    phases = np.asarray(summary["phases"], dtype=np.float64)

    model = IdealCZModel(species=idealised_yb171())
    config = PhaseOnlyGrapeConfig(
        num_tslots=len(phases),
        evo_time=4.0,
        max_iter=200,
        leakage_weight=5.0,
        seed=7,
        init_phase_spread=0.2,
    )
    optimizer = PhaseOnlyGrapeOptimizer(model=model, config=config)

    times_01, states_01 = optimizer.trajectory(phases, computational_basis_state(model, "01"))
    times_11, states_11 = optimizer.trajectory(phases, computational_basis_state(model, "11"))
    pops_01 = state_populations(states_01)
    pops_11 = state_populations(states_11)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    labels = ["CZ Fidelity", "Family Fidelity", "Leakage"]
    values = [
        summary["fidelity_to_cz"],
        summary["fidelity_to_entangling_family"],
        summary["leakage"],
    ]
    colors = ["#2c7fb8", "#41ab5d", "#d95f0e"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Current Reference Metrics")
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.03, 0.97), f"{value:.3f}", ha="center", va="bottom")

    ax = axes[0, 1]
    ax.step(np.arange(len(phases)), phases, where="mid", color="#6a3d9a")
    ax.set_title("Optimized Phase Pulse")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Phase [rad]")
    ax.set_ylim(-0.1, 2.0 * np.pi + 0.1)

    ax = axes[1, 0]
    ax.plot(times_01, pops_01["01"], label="|01>", color="#1f78b4")
    ax.plot(times_01, pops_01["0r"], label="|0r>", color="#33a02c")
    ax.set_title("Trajectory from |01>")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times_11, pops_11["11"], label="|11>", color="#e31a1c")
    ax.plot(times_11, pops_11["W"], label="|W>", color="#ff7f00")
    ax.set_title("Trajectory from |11>")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    fig.suptitle("Frozen Reference Experiment: ideal 171Yb CZ")
    fig.tight_layout()

    output_path = ROOT / "artifacts" / "reference_2202_00903_ideal_yb_cz.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
