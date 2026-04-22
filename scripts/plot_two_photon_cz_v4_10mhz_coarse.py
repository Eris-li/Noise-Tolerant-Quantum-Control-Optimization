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


def wrapped_phase(phases: np.ndarray) -> np.ndarray:
    return (phases + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    artifacts = ROOT / "artifacts"
    coarse_summary = json.loads(
        (artifacts / "two_photon_cz_v4_0_300ns_10mhz_coarse_summary.json").read_text(encoding="utf-8")
    )
    points = coarse_summary["points"]
    optimized_points = [point for point in points if float(point.get("gate_time_ns", 0.0)) > 0.0 and "ctrl_x" in point]
    if not optimized_points:
        raise RuntimeError("No optimized coarse-scan points found.")

    best = max(optimized_points, key=lambda point: float(point["probe_fidelity"]))
    gate_times = [float(point["gate_time_ns"]) for point in points]
    fidelities = [float(point["probe_fidelity"]) for point in points]
    threshold = 0.99
    first_threshold_point = next(
        (point for point in optimized_points if float(point["probe_fidelity"]) >= threshold),
        None,
    )

    ctrl_x = np.asarray(best["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(best["ctrl_y"], dtype=np.float64)
    amplitudes = np.asarray(best["effective_rabi_sequence_mhz"], dtype=np.float64)
    phases = np.asarray(best["phases"], dtype=np.float64)
    centered = centered_phase(phases)
    wrapped = wrapped_phase(phases)
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
        ),
    )

    probe_plus = optimizer.model.probe_kets(theta=0.0)[2][0]
    times, states = optimizer.trajectory(ctrl_x, ctrl_y, qutip.ket2dm(probe_plus))
    array = np.stack([state.full() for state in states], axis=0)
    basis_labels = list(optimizer.model.basis_labels())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(gate_times, fidelities, marker="o", linewidth=2.0, markersize=5, color="#1f78b4")
    ax.axhline(threshold, color="#d95f0e", linestyle="--", label="F = 0.99")
    if first_threshold_point is not None:
        ax.axvline(
            float(first_threshold_point["gate_time_ns"]),
            color="#6a3d9a",
            linestyle=":",
            label=f"first F≥0.99: T={first_threshold_point['gate_time_ns']:.1f} ns",
        )
    ax.set_xlabel("UV Segment Time (ns)")
    ax.set_ylabel("Phase-Gate Fidelity")
    ax.set_ylim(0.35, 1.001)
    ax.set_title("v4 10 MHz coarse scan of full-gate model")
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
    ax.set_title("Optimized amplitude and phase")

    ax = axes[1, 0]
    smooth_x = np.arange(1, len(amplitudes))
    ax.plot(smooth_x, amp_smooth, color="#e31a1c", label="Δ amplitude [MHz]")
    ax.plot(smooth_x, phase_smooth, color="#1f78b4", label="Δ phase [rad]")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("First difference")
    ax.set_title("Smoothness diagnostics")
    ax.legend()

    pop_spec = axes[1, 1].get_subplotspec()
    axes[1, 1].remove()
    pop_gs = pop_spec.subgridspec(2, 1, hspace=0.28)
    ax = fig.add_subplot(pop_gs[0, 0])
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
    clock_indices = [
        index for index, label in enumerate(basis_labels) if label in {"0c", "W_c", "cc"}
    ]
    mixed_cr_indices = [index for index, label in enumerate(basis_labels) if label == "W_cr"]
    rr_indices = [index for index, label in enumerate(basis_labels) if label == "rr"]
    leak_indices = [index for index, label in enumerate(basis_labels) if "leak" in label]
    loss_indices = [index for index, label in enumerate(basis_labels) if "loss" in label]
    rydberg_indices = [
        index
        for index, label in enumerate(basis_labels)
        if label in {"0r", "W_r"}
    ]

    ax.plot(times_ns, pop_sum(active_01), label="|01>", color="#1f78b4")
    ax.plot(times_ns, pop_sum(active_11), label="|11>", color="#e31a1c")
    ax.plot(times_ns, pop_sum(clock_indices), label="clock-shelved total", color="#33a02c")
    ax.plot(times_ns, pop_sum(rydberg_indices), label="single-Rydberg total", color="#6a3d9a")
    ax.plot(times_ns, pop_sum(mixed_cr_indices), label="mixed |cr>+|rc>", color="#1b9e77")
    ax.plot(times_ns, pop_sum(rr_indices), label="|rr>", color="#b15928")
    if leak_indices:
        ax.plot(times_ns, pop_sum(leak_indices), label="|leak>", color="#a6761d")
    if loss_indices:
        ax.plot(times_ns, pop_sum(loss_indices), label="|loss>", color="#ff7f00")
    ax.set_xlabel("Total Gate Time (ns)")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Full-gate population dynamics")
    ax.legend()

    uv_start_ns = float(best["clock_pi_pulse_duration_ns"])
    uv_end_ns = uv_start_ns + float(best["gate_time_ns"])
    uv_mask = (times_ns >= uv_start_ns) & (times_ns <= uv_end_ns)
    uv_times_ns = times_ns[uv_mask] - uv_start_ns

    ax = fig.add_subplot(pop_gs[1, 0])
    ax.plot(uv_times_ns, pop_sum(active_01)[uv_mask], label="|01>", color="#1f78b4")
    ax.plot(uv_times_ns, pop_sum(active_11)[uv_mask], label="|11>", color="#e31a1c")
    ax.plot(uv_times_ns, pop_sum(clock_indices)[uv_mask], label="clock-shelved total", color="#33a02c")
    ax.plot(uv_times_ns, pop_sum(rydberg_indices)[uv_mask], label="single-Rydberg total", color="#6a3d9a")
    ax.plot(uv_times_ns, pop_sum(mixed_cr_indices)[uv_mask], label="mixed |cr>+|rc>", color="#1b9e77")
    ax.plot(uv_times_ns, pop_sum(rr_indices)[uv_mask], label="|rr>", color="#b15928")
    if leak_indices:
        ax.plot(uv_times_ns, pop_sum(leak_indices)[uv_mask], label="|leak>", color="#a6761d")
    if loss_indices:
        ax.plot(uv_times_ns, pop_sum(loss_indices)[uv_mask], label="|loss>", color="#ff7f00")
    ax.set_xlabel("UV Segment Time (ns)")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("UV-segment population dynamics")
    ax.legend()

    fig.suptitle("^171Yb v4 full-gate open-system coarse scan at 10 MHz")
    fig.tight_layout()

    output_path = artifacts / "two_photon_cz_v4_0_300ns_10mhz_summary.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    pulse_fig, pulse_axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    pulse_x = np.arange(len(amplitudes))

    pulse_axes[0].plot(pulse_x, amplitudes, color="#e31a1c", linewidth=1.8)
    pulse_axes[0].set_ylabel("Amplitude [MHz]")
    pulse_axes[0].set_ylim(0.0, 15.0)
    pulse_axes[0].set_title("Optimized amplitude")

    pulse_axes[1].plot(pulse_x, wrapped, color="#1f78b4", linewidth=1.8)
    pulse_axes[1].set_xlabel("Time slice")
    pulse_axes[1].set_ylabel("Phase [rad]")
    pulse_axes[1].set_ylim(-np.pi, np.pi)
    pulse_axes[1].set_title("Optimized phase")

    pulse_fig.suptitle("^171Yb v4 optimized UV control at coarse optimum")
    pulse_fig.tight_layout()
    pulse_output_path = artifacts / "two_photon_cz_v4_0_300ns_10mhz_optimized_pulse.png"
    pulse_fig.savefig(pulse_output_path, dpi=180)
    plt.close(pulse_fig)

    print(output_path)
    print(pulse_output_path)


if __name__ == "__main__":
    main()
