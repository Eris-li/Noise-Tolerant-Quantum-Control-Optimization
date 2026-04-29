from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ma2023_time_optimal_2q_dir  # noqa: E402
from neutral_yb.config.ma2023_calibration import (  # noqa: E402
    build_ma2023_six_level_model,
    load_ma2023_fig3_data,
    ma2023_experimental_calibration,
)
from neutral_yb.models.ma2023_pulse import Ma2023GaussianEdgePulse  # noqa: E402
from neutral_yb.optimization.ma2023_six_level_grape import (  # noqa: E402
    Ma2023SixLevelGRAPEConfig,
    Ma2023SixLevelPhaseOptimizer,
)


def parse_args() -> argparse.Namespace:
    default_dir = ma2023_time_optimal_2q_dir(ROOT) / "from_method"
    parser = argparse.ArgumentParser(description="Compare Ma 2023 six-level output with Dataverse Fig. 3 data.")
    parser.add_argument(
        "--result",
        type=Path,
        default=default_dir / "ma2023_six_level_cheb13_N100_phi0fixed_from_direct_ideal.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "ma2023_six_level_cheb13_N100_phi0fixed_comparison.png",
    )
    return parser.parse_args()


def transition_bloch_trajectory(
    optimizer: Ma2023SixLevelPhaseOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    sector: str,
) -> dict[str, np.ndarray]:
    indices = optimizer.model.transition_subspace_indices(sector)
    computational = indices[0]
    rydberg = indices[1:]
    state = np.zeros(optimizer.dimension, dtype=np.complex128)
    state[computational] = 1.0
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    def append_point(vector: np.ndarray) -> None:
        rydberg_amp = np.sqrt(sum(abs(vector[index]) ** 2 for index in rydberg))
        bright = sum(vector[index] for index in rydberg)
        if abs(bright) > 1e-12:
            bright /= abs(bright)
        rho_00 = abs(vector[computational]) ** 2
        rho_11 = rydberg_amp**2
        coh = np.conj(vector[computational]) * bright * rydberg_amp
        norm = rho_00 + rho_11
        if norm > 1e-12:
            xs.append(float(2.0 * np.real(coh) / norm))
            ys.append(float(2.0 * np.imag(coh) / norm))
            zs.append(float((rho_00 - rho_11) / norm))
        else:
            xs.append(0.0)
            ys.append(0.0)
            zs.append(0.0)

    append_point(state)
    for x_value, y_value in zip(ctrl_x, ctrl_y):
        state = expm(optimizer.config.dt * optimizer._generator(float(x_value), float(y_value))) @ state
        append_point(state)
    return {"x": np.asarray(xs), "y": np.asarray(ys), "z": np.asarray(zs)}


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    fig3 = load_ma2023_fig3_data(ROOT)
    calibration = ma2023_experimental_calibration(root=ROOT)
    model = build_ma2023_six_level_model(include_noise=False, calibration=calibration)
    optimizer = Ma2023SixLevelPhaseOptimizer(
        model=model,
        config=Ma2023SixLevelGRAPEConfig(
            num_tslots=len(result["amplitudes"]),
            evo_time=float(calibration.target_dimensionless_duration),
            max_iter=1,
            num_restarts=1,
        ),
        envelope=Ma2023GaussianEdgePulse(num_tslots=len(result["amplitudes"])).envelope(),
    )

    pulse = fig3["pulse"]
    t_data = np.asarray(pulse["time_us"], dtype=np.float64)
    amp_data = np.asarray(pulse["amplitude_fraction"], dtype=np.float64)
    phase_data = np.asarray(pulse["phase_rad"], dtype=np.float64)
    amplitudes = np.asarray(result["amplitudes"], dtype=np.float64)
    phases = np.asarray(result["optimized_phase_rad_bounded"], dtype=np.float64)
    ctrl_x = np.asarray(result["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(result["ctrl_y"], dtype=np.float64)
    duration_us = float(fig3["calibration"]["fig3_duration_us"])
    t_method = np.linspace(0.0, duration_us, amplitudes.size, endpoint=False)

    bloch_data_01 = fig3["fig3"]["bloch_01"]
    bloch_data_11 = fig3["fig3"]["bloch_11"]
    bloch_method_01 = transition_bloch_trajectory(optimizer, ctrl_x, ctrl_y, "01")
    bloch_method_11 = transition_bloch_trajectory(optimizer, ctrl_x, ctrl_y, "11")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(t_data, amp_data, color="0.65", label="Dataverse Fig. 3")
    axes[0, 0].plot(t_method, amplitudes, color="tab:blue", label="six-level Cheb-13")
    axes[0, 0].set_ylabel(r"$\Omega / \Omega_{\max}$")
    axes[0, 0].set_title("Pulse amplitude")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(t_data, phase_data, color="0.65", label="Dataverse Fig. 3")
    axes[0, 1].plot(t_method, phases, color="tab:orange", label="six-level Cheb-13")
    axes[0, 1].set_ylabel("phase [rad]")
    axes[0, 1].set_title("Pulse phase")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(bloch_data_01["x"], bloch_data_01["y"], color="0.72", linestyle="-", label="Dataverse |01>")
    axes[1, 0].plot(bloch_data_11["x"], bloch_data_11["y"], color="0.72", linestyle="--", label="Dataverse |11>")
    axes[1, 0].plot(bloch_method_01["x"], bloch_method_01["y"], color="tab:blue", linewidth=1.8, label="method |01>")
    axes[1, 0].plot(bloch_method_11["x"], bloch_method_11["y"], color="tab:orange", linewidth=1.8, label="method |11>")
    axes[1, 0].set_xlabel("Bloch x")
    axes[1, 0].set_ylabel("Bloch y")
    axes[1, 0].set_title("Rydberg-transition Bloch trajectory")
    axes[1, 0].legend(frameon=False)

    labels = ["target", "six-level F", "process F", "1 - leakage"]
    values = [
        float(fig3["calibration"]["target_two_qubit_fidelity"]),
        float(result["fidelity"]),
        float(result["process_fidelity"]),
        1.0 - float(result["leakage"]),
    ]
    axes[1, 1].bar(labels, values, color=["0.55", "tab:purple", "tab:red", "tab:green"])
    axes[1, 1].set_ylim(0.95, 1.0)
    axes[1, 1].set_ylabel("fidelity / survival")
    axes[1, 1].set_title("Fidelity comparison")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
