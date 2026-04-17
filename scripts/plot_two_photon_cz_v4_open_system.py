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

from neutral_yb.config.yb171_calibration import build_yb171_v4_calibrated_model
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def centered_phase(phases: np.ndarray) -> np.ndarray:
    wrapped = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return np.unwrap(wrapped)

def main() -> None:
    artifacts = ROOT / "artifacts"
    coarse = json.loads((artifacts / "two_photon_cz_v4_open_system_coarse.json").read_text(encoding="utf-8"))
    fine = json.loads((artifacts / "two_photon_cz_v4_open_system_fine.json").read_text(encoding="utf-8"))
    optimal = json.loads((artifacts / "two_photon_cz_v4_open_system_optimal.json").read_text(encoding="utf-8"))
    fit = json.loads((artifacts / "two_photon_cz_v4_open_system_fit.json").read_text(encoding="utf-8"))

    ctrl_x = np.asarray(optimal["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(optimal["ctrl_y"], dtype=np.float64)
    amplitudes = np.asarray(optimal["amplitudes"], dtype=np.float64)
    phases = np.asarray(optimal["phases"], dtype=np.float64)

    optimizer = OpenSystemGRAPEOptimizer(
        model=build_yb171_v4_calibrated_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=len(ctrl_x),
            evo_time=float(optimal["evo_time"]),
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
        ),
    )
    probe_plus = optimizer.model.probe_kets(theta=0.0)[2][0]
    times, states = optimizer.trajectory(ctrl_x, ctrl_y, qutip.ket2dm(probe_plus))
    array = np.stack([state.full() for state in states], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(coarse["durations"], coarse["fidelities"], marker="o", linewidth=2.0, markersize=5, color="#1f78b4", label="coarse")
    ax.axhline(0.999, color="#d95f0e", linestyle="--", label="F = 0.999")
    ax.axvline(optimal["evo_time"], color="#6a3d9a", linestyle=":", label=f"threshold point {optimal['evo_time']:.3f}")
    ax.axvline(fit["t_star"], color="#000000", linestyle="-.", label=f"fit T*={fit['t_star']:.3f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Probe Fidelity")
    ax.set_ylim(0.45, 1.001)
    ax.set_title("Open-system coarse scan")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(fine["durations"], fine["fidelities"], marker=".", linewidth=1.5, markersize=6, color="#33a02c", label="fine")
    ax.axhline(0.999, color="#d95f0e", linestyle="--", label="F = 0.999")
    ax.axvline(optimal["evo_time"], color="#6a3d9a", linestyle=":", label=f"T*={optimal['evo_time']:.3f}")
    ax.axvline(fit["t_star"], color="#000000", linestyle="-.", label=f"fit T*={fit['t_star']:.3f}")
    ax.set_xlabel("T * Omega_max")
    ax.set_ylabel("Probe Fidelity")
    ax.set_ylim(0.995, 1.0002)
    ax.set_title("Open-system fine scan")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.arange(len(amplitudes)), amplitudes, color="#1f78b4", label="amplitude")
    ax.plot(np.arange(len(phases)), centered_phase(phases), color="#6a3d9a", label="phase")
    ax.set_xlabel("Time slice")
    ax.set_title("Lower-leg amplitude and phase")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, np.real(array[:, 0, 0]), label="|01>", color="#1f78b4")
    ax.plot(times, np.real(array[:, 3, 3]), label="|11>", color="#e31a1c")
    ax.plot(times, np.real(array[:, 1, 1]), label="|0e>", color="#33a02c")
    ax.plot(times, np.real(array[:, 6, 6]), label="|W_r>", color="#ff7f00")
    ax.plot(times, np.real(array[:, 9, 9]), label="|loss>", color="#6a3d9a")
    ax.set_xlabel("t Omega_max")
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Population dynamics for |+> probe")
    ax.legend()

    fig.suptitle("Two-photon CZ v4 open-system two-stage scan")
    fig.tight_layout()

    output_path = artifacts / "two_photon_cz_v4_open_system_summary.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
