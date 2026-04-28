from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ma2023_time_optimal_2q_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ma 2023 time-optimal 2Q scan outputs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ma2023_time_optimal_2q_dir(ROOT) / "ma2023_open_system_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ma2023_time_optimal_2q_dir(ROOT) / "ma2023_open_system_summary.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    results = payload["results"]
    durations = np.asarray([item["gate_time_dimensionless"] for item in results], dtype=np.float64)
    fidelities = np.asarray([item["objective_fidelity"] for item in results], dtype=np.float64)
    erasure = np.asarray([item["mean_erasure_population"] for item in results], dtype=np.float64)
    active = np.asarray([item["mean_active_population"] for item in results], dtype=np.float64)

    best_file = args.summary.parent / payload["best_result_file"]
    best = json.loads(best_file.read_text(encoding="utf-8"))
    amplitudes = np.asarray(best["amplitudes"], dtype=np.float64)
    phases = np.asarray(best["phases"], dtype=np.float64)
    pulse_grid = np.linspace(0.0, 1.0, amplitudes.size, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(durations, fidelities, marker="o", label="simulated")
    axes[0].axhline(payload["target_two_qubit_fidelity"], color="tab:red", linestyle="--", label="Nature 0.980")
    axes[0].set_xlabel(r"$T\Omega_{\max}$")
    axes[0].set_ylabel("gate fidelity")
    axes[0].set_title("Time scan")
    axes[0].legend(frameon=False)

    axes[1].plot(durations, 1.0 - active, marker="o", label="non-active")
    axes[1].plot(durations, erasure, marker="s", label="erasure/leak")
    axes[1].set_xlabel(r"$T\Omega_{\max}$")
    axes[1].set_ylabel("population")
    axes[1].set_title("Loss channels")
    axes[1].legend(frameon=False)

    axes[2].plot(pulse_grid, amplitudes, label="amplitude")
    axes[2].plot(pulse_grid, phases / (2.0 * np.pi), label=r"phase / $2\pi$")
    axes[2].set_xlabel("normalized pulse time")
    axes[2].set_title("Best pulse")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
