from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt


def main() -> None:
    artifact_path = ROOT / "artifacts" / "reference_2202_00903_global_cz_scan.json"
    scan = json.loads(artifact_path.read_text(encoding="utf-8"))

    durations = scan["durations"]
    fidelities = scan["fidelities"]
    best_duration = scan["best_duration"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(durations, fidelities, marker="o", color="#2c7fb8")
    ax.axhline(0.999, color="#d95f0e", linestyle="--", label="target fidelity 0.999")
    if best_duration is not None:
        ax.axvline(best_duration, color="#41ab5d", linestyle=":", label=f"best duration {best_duration:.2f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Best phase-gate fidelity")
    ax.set_ylim(0.95, 1.001)
    ax.set_title("Time-optimal scan for paper-style global CZ")
    ax.legend()
    fig.tight_layout()

    output_path = ROOT / "artifacts" / "reference_2202_00903_global_cz_scan.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
