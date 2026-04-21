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

    ctrl_x = np.asarray(best["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(best["ctrl_y"], dtype=np.float64)
    amplitudes = np.asarray(best["effective_rabi_sequence_mhz"], dtype=np.float64)
    phases = np.asarray(best["phases"], dtype=np.float64)

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

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(gate_times, fidelities, marker="o", linewidth=2.0, markersize=5, color="#1f78b4")
    ax.axhline(0.999, color="#d95f0e", linestyle="--", label="F = 0.999")
    ax.axvline(float(best["gate_time_ns"]), color="#6a3d9a", linestyle=":", label=f"T={best['gate_time_ns']:.1f} ns")
    ax.set_xlabel("Gate Time (ns)")
    ax.set_ylabel("Phase-Gate Fidelity")
    ax.set_ylim(0.35, 1.001)
    ax.set_title("v4 10 MHz coarse time scan")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(phases)), centered_phase(phases), color="#1f78b4", label="optimized phase")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Phase [rad]")
    ax.set_title("Optimized phase sequence")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.arange(len(amplitudes)), amplitudes, color="#e31a1c", label="effective Rabi [MHz]")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Amplitude [MHz]")
    ax.set_title("Optimized amplitude sequence")
    ax.legend()

    ax = axes[1, 1]
    times_ns = np.asarray(
        [
            yb171_dimensionless_time_to_gate_time_ns(value, effective_rabi_hz=float(best["omega_max_hz"]))
            for value in times
        ],
        dtype=np.float64,
    )
    ax.plot(times_ns, np.real(array[:, 0, 0]), label="|01>", color="#1f78b4")
    ax.plot(times_ns, np.real(array[:, 3, 3]), label="|11>", color="#e31a1c")
    ax.plot(times_ns, np.real(array[:, 1, 1]) + np.real(array[:, 4, 4]) + np.real(array[:, 5, 5]), label="intermediate total", color="#33a02c")
    ax.plot(times_ns, np.real(array[:, 2, 2]) + np.real(array[:, 6, 6]) + np.real(array[:, 7, 7]) + np.real(array[:, 8, 8]), label="rydberg total", color="#6a3d9a")
    ax.plot(times_ns, np.real(array[:, 9, 9]), label="|loss>", color="#ff7f00")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population dynamics at coarse optimum")
    ax.legend()

    fig.suptitle("Two-photon CZ v4 open-system coarse scan at 10 MHz")
    fig.tight_layout()

    output_path = artifacts / "two_photon_cz_v4_0_300ns_10mhz_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
