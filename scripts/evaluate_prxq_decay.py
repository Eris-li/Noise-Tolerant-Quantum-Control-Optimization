from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-neutral-yb")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutral_yb.analysis.noise_tolerant_rydberg import blockade_branch_hamiltonians


ARTIFACT_DIR = ROOT / "notebooks" / "artifacts" / "prxq2023_smooth_4h"
OUTPUT_PATH = ARTIFACT_DIR / "decay_summary.json"
CURVES_PATH = ARTIFACT_DIR / "decay_sensitivity_curves.json"
FIGURE_PATH = ARTIFACT_DIR / "decay_sensitivity_curves.png"


def normalize_phases(phases: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(phases, dtype=np.float64), 2.0 * np.pi)


def make_repeated_half_pulse(half_phases: np.ndarray) -> np.ndarray:
    return np.concatenate([normalize_phases(half_phases), normalize_phases(half_phases)])


def diagonal_cz_fidelity(alpha10: complex, alpha01: complex, beta11: complex, theta10: float, theta01: float) -> float:
    overlap = 1.0 + np.exp(-1j * theta10) * alpha10 + np.exp(-1j * theta01) * alpha01
    overlap -= np.exp(-1j * (theta10 + theta01)) * beta11
    return float(abs(overlap) ** 2 / 16.0)


def optimize_local_phases(alpha10: complex, alpha01: complex, beta11: complex) -> tuple[float, float, float]:
    def objective(variables: np.ndarray) -> float:
        return -diagonal_cz_fidelity(alpha10, alpha01, beta11, float(variables[0]), float(variables[1]))

    starts = (np.array([np.angle(alpha10), np.angle(alpha01)]), np.zeros(2), np.pi * np.ones(2))
    best = None
    for start in starts:
        result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 400})
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("local phase optimization did not produce a result")
    return float(np.mod(best.x[0], 2.0 * np.pi)), float(np.mod(best.x[1], 2.0 * np.pi)), -float(best.fun)


def branch_decay_diagonals(gamma_over_omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = float(gamma_over_omega)
    return (
        np.diag([0.0, gamma]).astype(np.complex128),
        np.diag([0.0, gamma]).astype(np.complex128),
        np.diag([0.0, gamma, gamma]).astype(np.complex128),
    )


def propagate_blockade_branches_with_decay(
    phases: np.ndarray,
    *,
    total_time: float,
    gamma_over_omega: float,
    amplitude_error: float = 0.0,
    detuning_1: float = 0.0,
    detuning_2: float = 0.0,
    use_doppler_reversal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = normalize_phases(phases)
    signs = np.ones(phases.size, dtype=np.float64)
    if use_doppler_reversal:
        signs[phases.size // 2 :] = -1.0
    decay_diagonals = branch_decay_diagonals(float(gamma_over_omega))
    states = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0, 0.0], dtype=np.complex128),
    ]
    dt = float(total_time) / phases.size
    for index, phase in enumerate(phases):
        hamiltonians = blockade_branch_hamiltonians(
            float(phase),
            amplitude_error=float(amplitude_error),
            detuning_1=float(detuning_1) * float(signs[index]),
            detuning_2=float(detuning_2) * float(signs[index]),
        )
        for branch_index, (hamiltonian, decay_diagonal) in enumerate(zip(hamiltonians, decay_diagonals, strict=True)):
            generator = (-1j * hamiltonian - 0.5 * decay_diagonal) * dt
            states[branch_index] = expm(generator) @ states[branch_index]
    return states[0], states[1], states[2]


def evaluate_full_gate_with_decay(
    phases: np.ndarray,
    *,
    total_time: float,
    gamma_over_omega: float,
    amplitude_error: float = 0.0,
    detuning_1: float = 0.0,
    detuning_2: float = 0.0,
    use_doppler_reversal: bool = False,
) -> dict[str, float]:
    states = propagate_blockade_branches_with_decay(
        phases,
        total_time=float(total_time),
        gamma_over_omega=float(gamma_over_omega),
        amplitude_error=float(amplitude_error),
        detuning_1=float(detuning_1),
        detuning_2=float(detuning_2),
        use_doppler_reversal=bool(use_doppler_reversal),
    )
    alpha10, alpha01, beta11 = complex(states[0][0]), complex(states[1][0]), complex(states[2][0])
    theta10, theta01, process_fidelity = optimize_local_phases(alpha10, alpha01, beta11)
    computational_return_population = (1.0 + abs(alpha10) ** 2 + abs(alpha01) ** 2 + abs(beta11) ** 2) / 4.0
    active_survival = (1.0 + sum(float(np.vdot(state, state).real) for state in states)) / 4.0
    return {
        "process_fidelity": float(process_fidelity),
        "infidelity": float(1.0 - process_fidelity),
        "theta10": float(theta10),
        "theta01": float(theta01),
        "computational_return_population": float(computational_return_population),
        "active_survival": float(active_survival),
        "decay_loss_proxy": float(1.0 - active_survival),
        "coherent_leakage_population": float(max(active_survival - computational_return_population, 0.0)),
    }


def load_current_candidates() -> dict[str, dict[str, Any]]:
    to_ar = np.load(ARTIFACT_DIR / "to_ar_phases.npz", allow_pickle=False)
    dr = np.load(ARTIFACT_DIR / "dr_half_phases.npz", allow_pickle=False)
    adr = np.load(ARTIFACT_DIR / "adr_half_phases.npz", allow_pickle=False)
    return {
        "TO optimized": {
            "phases": np.asarray(to_ar["to_phases"], dtype=np.float64),
            "total_time": float(to_ar["to_time"]),
            "use_doppler_reversal": False,
        },
        "AR full CZ": {
            "phases": np.asarray(to_ar["ar_phases"], dtype=np.float64),
            "total_time": float(to_ar["full_ar_time"]),
            "use_doppler_reversal": False,
        },
        "DR repeated half": {
            "phases": make_repeated_half_pulse(np.asarray(dr["dr_half_phases"], dtype=np.float64)),
            "total_time": 2.0 * float(dr["dr_half_total_time"]),
            "use_doppler_reversal": True,
        },
        "ADR repeated half": {
            "phases": make_repeated_half_pulse(np.asarray(adr["adr_half_phases"], dtype=np.float64)),
            "total_time": 2.0 * float(adr["adr_half_total_time"]),
            "use_doppler_reversal": True,
        },
    }


def main() -> None:
    omega_max_hz = 10.0e6
    rydberg_lifetime_s = 65.0e-6
    gamma_over_omega = 1.0 / (2.0 * math.pi * omega_max_hz * rydberg_lifetime_s)

    closed_summary = json.loads((ARTIFACT_DIR / "available_summary.json").read_text())
    closed_by_label = {row["label"]: row for row in closed_summary["summary_rows"]}
    candidates = load_current_candidates()
    rows: list[dict[str, float | str | bool]] = []
    for label, candidate in candidates.items():
        phases = np.asarray(candidate["phases"], dtype=np.float64)
        total_time = float(candidate["total_time"])
        use_doppler_reversal = bool(candidate["use_doppler_reversal"])
        nominal = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=gamma_over_omega,
            use_doppler_reversal=use_doppler_reversal,
        )
        amp_min = min(
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=gamma_over_omega,
                amplitude_error=value,
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for value in (-0.05, 0.05)
        )
        detuning_minus = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=gamma_over_omega,
            detuning_1=-0.05,
            use_doppler_reversal=use_doppler_reversal,
        )["process_fidelity"]
        detuning_plus = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=gamma_over_omega,
            detuning_1=0.05,
            use_doppler_reversal=use_doppler_reversal,
        )["process_fidelity"]
        closed = closed_by_label[label]
        rows.append(
            {
                "label": label,
                "total_time_omega": float(total_time),
                "physical_time_us": float(total_time / (2.0 * math.pi * omega_max_hz) * 1e6),
                "doppler_reversal": use_doppler_reversal,
                "closed_nominal_fidelity": float(closed["nominal_fidelity"]),
                "decay_nominal_fidelity": float(nominal["process_fidelity"]),
                "nominal_decay_delta": float(nominal["process_fidelity"] - float(closed["nominal_fidelity"])),
                "decay_active_survival": float(nominal["active_survival"]),
                "decay_loss_proxy": float(nominal["decay_loss_proxy"]),
                "decay_min_amp_fidelity_pm_0p05": float(amp_min),
                "decay_detuning_fidelity_m0p05": float(detuning_minus),
                "decay_detuning_fidelity_p0p05": float(detuning_plus),
            }
        )

    payload = {
        "noise_model": "Rydberg decay as branch-level non-Hermitian no-jump term; no pulse reoptimization",
        "omega_max_hz": omega_max_hz,
        "rydberg_lifetime_s": rydberg_lifetime_s,
        "gamma_over_omega": gamma_over_omega,
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))

    epsilon_grid = np.linspace(-0.10, 0.10, 81)
    detuning_grid = np.linspace(-0.10, 0.10, 81)
    curve_payload: dict[str, Any] = {
        "noise_model": payload["noise_model"],
        "omega_max_hz": omega_max_hz,
        "rydberg_lifetime_s": rydberg_lifetime_s,
        "gamma_over_omega": gamma_over_omega,
        "epsilon_grid": epsilon_grid.tolist(),
        "detuning_grid": detuning_grid.tolist(),
        "curves": {},
    }
    for label, candidate in candidates.items():
        phases = np.asarray(candidate["phases"], dtype=np.float64)
        total_time = float(candidate["total_time"])
        use_doppler_reversal = bool(candidate["use_doppler_reversal"])
        decay_amp = [
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=gamma_over_omega,
                amplitude_error=float(epsilon),
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for epsilon in epsilon_grid
        ]
        decay_det = [
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=gamma_over_omega,
                detuning_1=float(detuning),
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for detuning in detuning_grid
        ]
        closed_amp = [
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=0.0,
                amplitude_error=float(epsilon),
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for epsilon in epsilon_grid
        ]
        closed_det = [
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=0.0,
                detuning_1=float(detuning),
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for detuning in detuning_grid
        ]
        curve_payload["curves"][label] = {
            "decay_amplitude_fidelity": decay_amp,
            "decay_detuning_fidelity": decay_det,
            "closed_amplitude_fidelity": closed_amp,
            "closed_detuning_fidelity": closed_det,
        }
    CURVES_PATH.write_text(json.dumps(curve_payload, indent=2, sort_keys=True))

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex="col")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (label, curves) in zip(color_cycle, curve_payload["curves"].items(), strict=False):
        decay_amp = np.asarray(curves["decay_amplitude_fidelity"], dtype=np.float64)
        decay_det = np.asarray(curves["decay_detuning_fidelity"], dtype=np.float64)
        closed_amp = np.asarray(curves["closed_amplitude_fidelity"], dtype=np.float64)
        closed_det = np.asarray(curves["closed_detuning_fidelity"], dtype=np.float64)
        axes[0, 0].plot(epsilon_grid, decay_amp, color=color, label=label)
        axes[0, 0].plot(epsilon_grid, closed_amp, color=color, linestyle="--", alpha=0.45)
        axes[0, 1].plot(detuning_grid, decay_det, color=color, label=label)
        axes[0, 1].plot(detuning_grid, closed_det, color=color, linestyle="--", alpha=0.45)
        axes[1, 0].semilogy(epsilon_grid, np.maximum(1.0 - decay_amp, 1e-12), color=color, label=label)
        axes[1, 0].semilogy(epsilon_grid, np.maximum(1.0 - closed_amp, 1e-12), color=color, linestyle="--", alpha=0.45)
        axes[1, 1].semilogy(detuning_grid, np.maximum(1.0 - decay_det, 1e-12), color=color, label=label)
        axes[1, 1].semilogy(detuning_grid, np.maximum(1.0 - closed_det, 1e-12), color=color, linestyle="--", alpha=0.45)

    axes[0, 0].set_title("Amplitude error with Rydberg decay")
    axes[0, 1].set_title("Detuning error with Rydberg decay")
    axes[1, 0].set_title("Amplitude infidelity")
    axes[1, 1].set_title("Detuning infidelity")
    axes[1, 0].set_xlabel(r"Common amplitude error $\epsilon$")
    axes[1, 1].set_xlabel(r"$\Delta_1/\Omega_{\max}$")
    axes[0, 0].set_ylabel("Process fidelity")
    axes[1, 0].set_ylabel("Infidelity")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="lower center", fontsize=8)
    axes[1, 1].text(
        0.02,
        0.03,
        "solid: with decay; dashed: no decay",
        transform=axes[1, 1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {CURVES_PATH}")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
