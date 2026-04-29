from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ensure_artifact_dir, ma2023_time_optimal_2q_dir  # noqa: E402
from neutral_yb.config.ma2023_calibration import build_ma2023_six_level_model, ma2023_experimental_calibration  # noqa: E402
from neutral_yb.models.ma2023_noise import (  # noqa: E402
    Ma2023NoiseTraceConfig,
    doppler_detuning_rms_from_t2_star,
    generate_noise_trace,
)
from neutral_yb.optimization.ma2023_six_level_grape import (  # noqa: E402
    Ma2023SixLevelGRAPEConfig,
    Ma2023SixLevelPhaseOptimizer,
)
from scipy.linalg import expm  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_pulse = (
        ma2023_time_optimal_2q_dir(ROOT)
        / "from_method"
        / "ma2023_six_level_cheb13_N100_refined_ideal.json"
    )
    parser = argparse.ArgumentParser(description="Evaluate a Ma 2023 six-level pulse with noisy Lindblad traces.")
    parser.add_argument("--pulse", type=Path, default=default_pulse)
    parser.add_argument("--num-traces", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--doppler-scale", type=float, default=1.0)
    parser.add_argument("--phase-noise-rms-rad", type=float, default=0.0)
    parser.add_argument("--intensity-noise-rms", type=float, default=0.0)
    parser.add_argument("--method", choices=("nojump", "lindblad"), default="nojump")
    parser.add_argument("--output", type=str, default="ma2023_six_level_noisy_eval.json")
    return parser.parse_args()


def computational_density(model, index: int) -> np.ndarray:
    rho = np.zeros((model.dimension(), model.dimension()), dtype=np.complex128)
    rho[index, index] = 1.0
    return rho


def noisy_channel_summary(
    optimizer: Ma2023SixLevelPhaseOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    theta0: float,
    theta1: float,
    traces,
) -> dict[str, float]:
    model = optimizer.model
    computational = model.computational_indices()
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    signs = np.array([1.0, 1.0, -1.0], dtype=np.complex128)
    target_conj = np.array(
        [
            np.exp(-1j * theta0),
            np.exp(-1j * (theta0 + theta1)),
            np.exp(-1j * (theta0 + 2.0 * theta1)),
        ],
        dtype=np.complex128,
    )
    trace_values = []
    population_values = []
    erasure_values = []
    undetected_values = []
    for trace in traces:
        diagonal_response = []
        population_sum = 0.0
        erasure_sum = 0.0
        undetected_sum = 0.0
        for basis_index, weight in zip(computational, weights):
            rho_t = optimizer.evolve_density_matrix(
                ctrl_x,
                ctrl_y,
                computational_density(model, basis_index),
                noise_trace=trace,
            )
            diagonal_response.append(rho_t[basis_index, basis_index])
            population_sum += float(weight * np.real(rho_t[basis_index, basis_index]))
            if model.erasure_index() is not None:
                erasure_sum += float(weight * np.real(rho_t[model.erasure_index(), model.erasure_index()]))
            if model.undetected_decay_index() is not None:
                undetected_sum += float(weight * np.real(rho_t[model.undetected_decay_index(), model.undetected_decay_index()]))
        channel_trace = np.sum(weights * signs * target_conj * np.asarray(diagonal_response))
        trace_values.append(channel_trace)
        population_values.append(population_sum / 4.0)
        erasure_values.append(erasure_sum / 4.0)
        undetected_values.append(undetected_sum / 4.0)

    process_fidelities = [float(abs(value) ** 2 / 16.0) for value in trace_values]
    active_populations = np.asarray(population_values, dtype=np.float64)
    leakage = 1.0 - active_populations
    process_fidelity = float(np.mean(process_fidelities))
    mean_leakage = float(np.mean(leakage))
    gate_fidelity = float((4.0 * process_fidelity + 1.0 - mean_leakage) / 5.0)
    return {
        "gate_fidelity": gate_fidelity,
        "gate_error": float(1.0 - gate_fidelity),
        "process_fidelity": process_fidelity,
        "leakage": mean_leakage,
        "active_population": float(np.mean(active_populations)),
        "detected_decay": float(np.mean(erasure_values)),
        "undetected_decay": float(np.mean(undetected_values)),
        "trace_count": int(len(traces)),
        "process_fidelity_std": float(np.std(process_fidelities)),
        "leakage_std": float(np.std(leakage)),
    }


def nojump_channel_summary(
    optimizer: Ma2023SixLevelPhaseOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    theta0: float,
    theta1: float,
    traces,
) -> dict[str, float]:
    model = optimizer.model
    computational = model.computational_indices()
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    signs = np.array([1.0, 1.0, -1.0], dtype=np.complex128)
    target_conj = np.array(
        [
            np.exp(-1j * theta0),
            np.exp(-1j * (theta0 + theta1)),
            np.exp(-1j * (theta0 + 2.0 * theta1)),
        ],
        dtype=np.complex128,
    )
    gamma_r = float(model.noise.rydberg_decay_rate)
    detected_fraction = float(model.noise.rydberg_decay_detected_fraction)
    undetected_fraction = 1.0 - detected_fraction
    rydberg_indices = []
    for sector in model.sector_labels():
        rydberg_indices.extend(model.transition_subspace_indices(sector)[1:])
    process_fidelities = []
    leakage_values = []
    erasure_values = []
    undetected_values = []
    for trace in traces:
        trace.validate(ctrl_x.size)
        from neutral_yb.models.ma2023_noise import apply_noise_trace_to_controls

        noisy_ctrl_x, noisy_ctrl_y = apply_noise_trace_to_controls(ctrl_x, ctrl_y, trace)
        z_values = []
        active_weighted_population = 0.0
        erasure_weighted = 0.0
        undetected_weighted = 0.0
        for basis_index, weight in zip(computational, weights):
            state = np.zeros(optimizer.dimension, dtype=np.complex128)
            state[basis_index] = 1.0
            erasure_probability = 0.0
            undetected_probability = 0.0
            for x_value, y_value, detuning in zip(noisy_ctrl_x, noisy_ctrl_y, trace.common_detuning):
                before_rydberg = float(np.sum(np.abs(state[rydberg_indices]) ** 2))
                generator = optimizer._generator(float(x_value), float(y_value))
                if detuning != 0.0:
                    generator = generator + 1j * float(detuning) * optimizer.rydberg_projector
                state = expm(optimizer.config.dt * generator) @ state
                after_rydberg = float(np.sum(np.abs(state[rydberg_indices]) ** 2))
                mean_rydberg = 0.5 * (before_rydberg + after_rydberg)
                decay_probability = gamma_r * optimizer.config.dt * mean_rydberg
                erasure_probability += detected_fraction * decay_probability
                undetected_probability += undetected_fraction * decay_probability
            z_values.append(state[basis_index])
            active_weighted_population += weight * float(abs(state[basis_index]) ** 2)
            erasure_weighted += weight * erasure_probability
            undetected_weighted += weight * undetected_probability
        channel_trace = np.sum(weights * signs * target_conj * np.asarray(z_values))
        process_fidelity = float(abs(channel_trace) ** 2 / 16.0)
        active_population = active_weighted_population / 4.0
        leakage = max(0.0, 1.0 - active_population)
        process_fidelities.append(process_fidelity)
        leakage_values.append(leakage)
        erasure_values.append(erasure_weighted / 4.0)
        undetected_values.append(undetected_weighted / 4.0)
    process_fidelity = float(np.mean(process_fidelities))
    leakage = float(np.mean(leakage_values))
    gate_fidelity = float((4.0 * process_fidelity + 1.0 - leakage) / 5.0)
    return {
        "gate_fidelity": gate_fidelity,
        "gate_error": float(1.0 - gate_fidelity),
        "process_fidelity": process_fidelity,
        "leakage": leakage,
        "active_population": float(1.0 - leakage),
        "detected_decay": float(np.mean(erasure_values)),
        "undetected_decay": float(np.mean(undetected_values)),
        "trace_count": int(len(traces)),
        "process_fidelity_std": float(np.std(process_fidelities)),
        "leakage_std": float(np.std(leakage_values)),
    }


def main() -> None:
    args = parse_args()
    pulse = json.loads(args.pulse.read_text(encoding="utf-8"))
    ctrl_x = np.asarray(pulse["ctrl_x"], dtype=np.float64)
    ctrl_y = np.asarray(pulse["ctrl_y"], dtype=np.float64)
    calibration = ma2023_experimental_calibration(root=ROOT)
    model = build_ma2023_six_level_model(include_noise=True, calibration=calibration)
    optimizer = Ma2023SixLevelPhaseOptimizer(
        model=model,
        config=Ma2023SixLevelGRAPEConfig(
            num_tslots=ctrl_x.size,
            evo_time=float(calibration.target_dimensionless_duration),
            max_iter=1,
            num_restarts=1,
        ),
        envelope=np.sqrt(ctrl_x**2 + ctrl_y**2),
    )
    doppler_rms = args.doppler_scale * doppler_detuning_rms_from_t2_star(
        t2_star_s=float(calibration.doppler_t2_star_s),
        omega_ref_rad_s=calibration.omega_ref_rad_s(),
    )
    traces = [
        generate_noise_trace(
            Ma2023NoiseTraceConfig(
                num_tslots=ctrl_x.size,
                quasistatic_detuning_rms=doppler_rms,
                intensity_noise_rms_fractional=float(args.intensity_noise_rms),
                phase_noise_rms_rad=float(args.phase_noise_rms_rad),
                seed=int(args.seed + index),
            )
        )
        for index in range(int(args.num_traces))
    ]
    summary_fn = nojump_channel_summary if args.method == "nojump" else noisy_channel_summary
    summary = summary_fn(optimizer, ctrl_x, ctrl_y, float(pulse.get("theta0", 0.0)), float(pulse.get("theta1", 0.0)), traces)
    payload = {
        "pulse": str(args.pulse),
        "noise_model": {
            "rydberg_lifetime_s": float(calibration.rydberg_lifetime_s),
            "rydberg_decay_detected_fraction": float(calibration.rydberg_decay_detected_fraction),
            "doppler_t2_star_s": float(calibration.doppler_t2_star_s),
            "doppler_detuning_rms_dimensionless": float(doppler_rms),
            "phase_noise_rms_rad": float(args.phase_noise_rms_rad),
            "intensity_noise_rms_fractional": float(args.intensity_noise_rms),
            "simulation_method": args.method,
        },
        "summary": summary,
    }
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT) / "from_method")
    output_path = output_dir / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
