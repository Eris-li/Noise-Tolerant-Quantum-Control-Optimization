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
import qutip

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    yb171_dimensionless_time_to_gate_time_ns,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)


def main() -> None:
    artifacts = ROOT / "artifacts"
    summary = json.loads(
        (artifacts / "two_photon_cz_v4_full_gate_300ns_10mhz_summary.json").read_text(encoding="utf-8")
    )
    best = json.loads(
        (artifacts / "two_photon_cz_v4_full_gate_300ns_10mhz_best.json").read_text(encoding="utf-8")
    )

    stage_summaries = summary["stage_summaries"]
    stage_labels = [item["stage"] for item in stage_summaries]
    stage_fidelities = [float(item["probe_fidelity"]) for item in stage_summaries]

    ctrl_x = np.asarray(best["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(best["ctrl_y"], dtype=np.float64)
    amplitudes = np.asarray(best["effective_rabi_sequence_mhz"], dtype=np.float64)
    phases = np.asarray(best["phases"], dtype=np.float64)
    centered = centered_phase(phases)
    amp_smooth = np.diff(amplitudes)
    phase_smooth = np.diff(centered)

    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(
            include_noise=True,
            effective_rabi_hz=float(best["omega_max_hz"]),
        ),
        config=OpenSystemGRAPEConfig(
            num_tslots=len(ctrl_x),
            evo_time=float(best["dimensionless_gate_time"]),
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            objective_metric="special_state",
        ),
    )

    probe_plus = optimizer.model.probe_kets(theta=0.0)[2][0]
    times, states = optimizer.trajectory(ctrl_x, ctrl_y, qutip.ket2dm(probe_plus))
    array = np.stack([state.full() for state in states], axis=0)
    basis_labels = list(optimizer.model.basis_labels())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(np.arange(len(stage_fidelities)), stage_fidelities, marker="o", linewidth=2.0, color="#1f78b4")
    ax.axhline(0.99, color="#d95f0e", linestyle="--", label="F = 0.99")
    ax.axhline(0.999, color="#6a3d9a", linestyle=":", label="F = 0.999")
    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels(stage_labels, rotation=20)
    ax.set_ylim(0.35, 1.001)
    ax.set_ylabel("Phase-Gate Fidelity")
    ax.set_title("300 ns optimization stages")
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(amplitudes))
    ax.plot(x, amplitudes, color="#e31a1c", label="effective Rabi [MHz]")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Amplitude [MHz]", color="#e31a1c")
    ax.tick_params(axis="y", labelcolor="#e31a1c")
    ax2 = ax.twinx()
    ax2.plot(x, centered, color="#1f78b4", label="phase [rad]")
    ax2.set_ylabel("Phase [rad]", color="#1f78b4")
    ax2.tick_params(axis="y", labelcolor="#1f78b4")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="best")
    ax.set_title("Best pulse at 300 ns")

    ax = axes[1, 0]
    smooth_x = np.arange(1, len(amplitudes))
    ax.plot(smooth_x, amp_smooth, color="#e31a1c", label="Δ amplitude [MHz]")
    ax.plot(smooth_x, phase_smooth, color="#1f78b4", label="Δ phase [rad]")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("First difference")
    ax.set_title("Smoothness diagnostics")
    ax.legend()

    ax = axes[1, 1]
    times_ns = np.asarray(
        [
            yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=float(best["omega_max_hz"]))
            for value in times
        ],
        dtype=np.float64,
    )
    populations = np.real(np.diagonal(array, axis1=1, axis2=2))

    def pop_sum(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(populations.shape[0], dtype=np.float64)
        return np.sum(populations[:, indices], axis=1)

    active_01 = [index for index, label in enumerate(basis_labels) if label == "01"]
    active_11 = [index for index, label in enumerate(basis_labels) if label == "11"]
    rr_indices = [index for index, label in enumerate(basis_labels) if label == "rr"]
    leak_indices = [index for index, label in enumerate(basis_labels) if "leak" in label]
    loss_indices = [index for index, label in enumerate(basis_labels) if "loss" in label]
    rydberg_indices = [
        index
        for index, label in enumerate(basis_labels)
        if "r" in label and label not in {"rr", "leak", "loss"}
    ]

    ax.plot(times_ns, pop_sum(active_01), label="|01>", color="#1f78b4")
    ax.plot(times_ns, pop_sum(active_11), label="|11>", color="#e31a1c")
    ax.plot(times_ns, pop_sum(rydberg_indices), label="single-Rydberg total", color="#33a02c")
    ax.plot(times_ns, pop_sum(rr_indices), label="|rr>", color="#6a3d9a")
    if leak_indices:
        ax.plot(times_ns, pop_sum(leak_indices), label="|leak>", color="#b15928")
    if loss_indices:
        ax.plot(times_ns, pop_sum(loss_indices), label="|loss>", color="#ff7f00")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population dynamics of best pulse")
    ax.legend()

    fig.suptitle("^171Yb v4 full-gate 300 ns optimization at 10 MHz")
    fig.tight_layout()

    output_path = artifacts / "two_photon_cz_v4_full_gate_300ns_10mhz_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
