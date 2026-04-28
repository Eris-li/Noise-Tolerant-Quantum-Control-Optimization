from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import qutip

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ensure_artifact_dir, ma2023_time_optimal_2q_dir  # noqa: E402
from neutral_yb.config.ma2023_calibration import (  # noqa: E402
    build_ma2023_model,
    build_ma2023_quasistatic_ensemble,
    load_ma2023_fig3_data,
    ma2023_experimental_calibration,
    ma2023_fig3_controls,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the imported Ma 2023 Fig. 3 pulse.")
    parser.add_argument("--num-tslots", type=int, default=96)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--output", type=str, default="ma2023_fig3_pulse_evaluation.json")
    return parser.parse_args()


def average_sink_populations(
    optimizer: OpenSystemGRAPEOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
) -> dict[str, float]:
    model = optimizer.model
    leak_index = int(model.leak_index())
    loss_index = int(model.loss_index())
    active_indices = tuple(int(index) for index in model.active_gate_indices())
    leakage_values: list[float] = []
    loss_values: list[float] = []
    active_values: list[float] = []
    for source_ket, _target in model.probe_kets(theta=0.0):
        final_rho = optimizer.evolve_density_matrix(ctrl_x, ctrl_y, qutip.ket2dm(source_ket))
        populations = np.real(np.diag(np.asarray(final_rho.full(), dtype=np.complex128)))
        leakage_values.append(float(populations[leak_index]))
        loss_values.append(float(populations[loss_index]))
        active_values.append(float(sum(populations[index] for index in active_indices)))
    return {
        "mean_active_population": float(np.mean(active_values)),
        "mean_leakage_population": float(np.mean(leakage_values)),
        "mean_loss_population": float(np.mean(loss_values)),
        "mean_erasure_population": float(np.mean(leakage_values) + np.mean(loss_values)),
    }


def main() -> None:
    args = parse_args()
    include_noise = not bool(args.no_noise)
    fig3_data = load_ma2023_fig3_data(ROOT)
    calibration = ma2023_experimental_calibration(root=ROOT)
    ctrl_x, ctrl_y, duration = ma2023_fig3_controls(num_tslots=args.num_tslots, root=ROOT)
    model = build_ma2023_model(include_noise=include_noise, calibration=calibration)
    ensemble = build_ma2023_quasistatic_ensemble(
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        include_noise=include_noise,
        calibration=calibration,
    )
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=args.num_tslots,
            evo_time=duration,
            max_iter=0,
            num_restarts=1,
            objective_metric="special_state",
            benchmark_active_channel=True,
        ),
        ensemble_models=ensemble,
    )
    theta_phase, phase_fidelity = optimizer.optimize_theta_for_phase_fidelity(ctrl_x, ctrl_y)
    theta_channel, active_channel_fidelity = optimizer.optimize_theta_for_channel(ctrl_x, ctrl_y)
    diagnostics = average_sink_populations(optimizer, ctrl_x, ctrl_y)
    payload = {
        "source": fig3_data["source"],
        "include_noise": bool(include_noise),
        "ensemble_size": int(args.ensemble_size),
        "num_tslots": int(args.num_tslots),
        "calibration": calibration.summary(),
        "phase_gate_theta": float(theta_phase),
        "phase_gate_fidelity": float(phase_fidelity),
        "active_channel_theta": float(theta_channel),
        "active_channel_fidelity": float(active_channel_fidelity),
        "target_two_qubit_fidelity": float(calibration.target_two_qubit_fidelity),
        "target_two_qubit_error": float(1.0 - calibration.target_two_qubit_fidelity),
        "simulated_phase_gate_error": float(1.0 - phase_fidelity),
        "simulated_active_channel_error": float(1.0 - active_channel_fidelity),
        "pulse_duration_dimensionless": float(duration),
        "pulse_peak_fraction": float(np.max(np.sqrt(ctrl_x**2 + ctrl_y**2))),
        **diagnostics,
    }
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT))
    output_path = output_dir / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()
