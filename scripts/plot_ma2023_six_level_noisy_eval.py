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
    parser = argparse.ArgumentParser(description="Plot Ma 2023 six-level noisy evaluation summary.")
    parser.add_argument("--eval", type=Path, default=default_dir / "ma2023_six_level_noisy_eval_100slot_4trace_nojump.json")
    parser.add_argument("--output", type=Path, default=default_dir / "ma2023_six_level_noisy_eval_100slot_4trace_nojump.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    pulse = json.loads((ROOT / payload["pulse"]).read_text(encoding="utf-8"))
    fig3 = load_ma2023_fig3_data(ROOT)
    summary = payload["summary"]
    noise = payload["noise_model"]

    duration_us = float(fig3["calibration"]["fig3_duration_us"])
    amplitudes = np.asarray(pulse["amplitudes"], dtype=np.float64)
    phases = np.asarray(pulse["optimized_phase_rad_bounded"], dtype=np.float64)
    t_method = np.linspace(0.0, duration_us, amplitudes.size, endpoint=False)
    t_data = np.asarray(fig3["pulse"]["time_us"], dtype=np.float64)
    amp_data = np.asarray(fig3["pulse"]["amplitude_fraction"], dtype=np.float64)
    phase_data = np.asarray(fig3["pulse"]["phase_rad"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    axes[0, 0].plot(t_data, amp_data, color="0.68", label="Dataverse Fig. 3")
    axes[0, 0].plot(t_method, amplitudes, color="tab:blue", linewidth=1.8, label="evaluated pulse")
    axes[0, 0].set_title("Pulse amplitude")
    axes[0, 0].set_ylabel(r"$\Omega / \Omega_{\max}$")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(t_data, phase_data, color="0.68", label="Dataverse Fig. 3")
    axes[0, 1].plot(t_method, phases, color="tab:orange", linewidth=1.8, label="evaluated pulse")
    axes[0, 1].set_title("Pulse phase")
    axes[0, 1].set_ylabel("phase [rad]")
    axes[0, 1].legend(frameon=False)

    fidelity_labels = ["paper target", "noisy gate", "process", "active pop."]
    fidelity_values = [
        float(fig3["calibration"]["target_two_qubit_fidelity"]),
        float(summary["gate_fidelity"]),
        float(summary["process_fidelity"]),
        float(summary["active_population"]),
    ]
    axes[1, 0].bar(fidelity_labels, fidelity_values, color=["0.55", "tab:purple", "tab:red", "tab:green"])
    axes[1, 0].set_ylim(max(0.90, min(fidelity_values) - 0.03), 1.0)
    axes[1, 0].set_title("Noisy fidelity summary")
    axes[1, 0].set_ylabel("fidelity / survival")
    axes[1, 0].tick_params(axis="x", rotation=18)

    error_labels = ["gate error", "leakage", "detected decay", "undetected decay"]
    error_values = [
        float(summary["gate_error"]),
        float(summary["leakage"]),
        float(summary["detected_decay"]),
        float(summary["undetected_decay"]),
    ]
    axes[1, 1].bar(error_labels, error_values, color=["tab:brown", "tab:gray", "tab:cyan", "tab:olive"])
    axes[1, 1].set_title("Error and leakage channels")
    axes[1, 1].set_ylabel("probability")
    axes[1, 1].tick_params(axis="x", rotation=18)

    caption = (
        f"method={noise['simulation_method']}, traces={summary['trace_count']}, "
        f"T1,r={1e6 * noise['rydberg_lifetime_s']:.1f} us, "
        f"T2*={1e6 * noise['doppler_t2_star_s']:.1f} us, "
        f"phase rms={noise['phase_noise_rms_rad']:.3g} rad, "
        f"intensity rms={noise['intensity_noise_rms_fractional']:.3g}"
    )
    fig.suptitle("Ma 2023 six-level noisy evaluation", y=0.99)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
