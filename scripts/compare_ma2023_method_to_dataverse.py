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
from neutral_yb.config.ma2023_calibration import load_ma2023_fig3_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_dir = ma2023_time_optimal_2q_dir(ROOT) / "from_method"
    parser = argparse.ArgumentParser(description="Compare method-first Ma 2023 output with Dataverse Fig. 3 data.")
    parser.add_argument("--summary", type=Path, default=default_dir / "ma2023_from_method_summary.json")
    parser.add_argument("--output", type=Path, default=default_dir / "ma2023_from_method_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    best = json.loads((args.summary.parent / summary["best_result_file"]).read_text(encoding="utf-8"))
    fig3 = load_ma2023_fig3_data(ROOT)

    pulse = fig3["pulse"]
    t_data = np.asarray(pulse["time_us"], dtype=np.float64)
    amp_data = np.asarray(pulse["amplitude_fraction"], dtype=np.float64)
    phase_data = np.asarray(pulse["phase_rad"], dtype=np.float64)

    amplitudes = np.asarray(best["amplitudes"], dtype=np.float64)
    phases = np.asarray(
        best.get("optimized_phase_rad_bounded", best["phases"]),
        dtype=np.float64,
    )
    duration_us = float(fig3["calibration"]["fig3_duration_us"])
    t_method = np.linspace(0.0, duration_us, amplitudes.size, endpoint=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(t_data, amp_data, color="0.65", label="Dataverse Fig. 3")
    axes[0, 0].plot(t_method, amplitudes, color="tab:blue", label="method-first")
    axes[0, 0].set_ylabel(r"$\Omega / \Omega_{\max}$")
    axes[0, 0].set_title("Pulse amplitude")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(t_data, phase_data, color="0.65", label="Dataverse Fig. 3")
    axes[0, 1].plot(t_method, phases, color="tab:orange", label="method-first")
    axes[0, 1].set_ylabel("phase [rad]")
    axes[0, 1].set_title("Pulse phase")
    axes[0, 1].legend(frameon=False)

    bloch_data_01 = fig3["fig3"]["bloch_01"]
    bloch_data_11 = fig3["fig3"]["bloch_11"]
    bloch_method_01 = best["bloch_01"]
    bloch_method_11 = best["bloch_11"]
    axes[1, 0].plot(
        bloch_data_01["x"],
        bloch_data_01["y"],
        color="0.72",
        linestyle="-",
        label="Dataverse |01>",
    )
    axes[1, 0].plot(
        bloch_data_11["x"],
        bloch_data_11["y"],
        color="0.72",
        linestyle="--",
        label="Dataverse |11>",
    )
    axes[1, 0].plot(
        bloch_method_01["x"],
        bloch_method_01["y"],
        color="tab:blue",
        linewidth=1.8,
        label="method |01>",
    )
    axes[1, 0].plot(
        bloch_method_11["x"],
        bloch_method_11["y"],
        color="tab:orange",
        linewidth=1.8,
        label="method |11>",
    )
    axes[1, 0].set_xlabel("Bloch x")
    axes[1, 0].set_ylabel("Bloch y")
    axes[1, 0].set_title("Rydberg-transition Bloch trajectory")
    axes[1, 0].legend(frameon=False)

    labels = ["target", "method phase", "method channel"]
    values = [
        float(summary["target_two_qubit_fidelity"]),
        float(best["probe_fidelity"]),
        float(best["active_channel_fidelity"]) if best["active_channel_fidelity"] is not None else np.nan,
    ]
    axes[1, 1].bar(labels, values, color=["0.55", "tab:purple", "tab:red"])
    axes[1, 1].set_ylim(0.95, 1.0)
    axes[1, 1].set_ylabel("fidelity")
    axes[1, 1].set_title("Fidelity comparison")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
