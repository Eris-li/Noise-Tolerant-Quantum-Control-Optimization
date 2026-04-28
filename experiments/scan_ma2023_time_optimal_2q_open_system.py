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
    ma2023_fig3_controls,
    ma2023_experimental_calibration,
    summarize_ma2023_result,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-system Ma 2023 time-optimal two-qubit-gate scan."
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=None,
        help="Dimensionless gate durations T * Omega_max.",
    )
    parser.add_argument(
        "--times-ns",
        nargs="+",
        type=float,
        default=None,
        help="Physical gate times in ns. Overrides --durations when set.",
    )
    parser.add_argument("--omega-max-mhz", type=float, default=None)
    parser.add_argument("--num-tslots", type=int, default=96)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--num-restarts", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--init-pulse-type", type=str, default="SINE")
    parser.add_argument("--init-control-scale", type=float, default=0.75)
    parser.add_argument("--objective-metric", type=str, default="special_state")
    parser.add_argument("--fidelity-target", type=float, default=0.98)
    parser.add_argument("--initial-pulse-from-data", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="ma2023_open_system")
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
        matrix = np.asarray(final_rho.full(), dtype=np.complex128)
        populations = np.real(np.diag(matrix))
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
    calibration = ma2023_experimental_calibration(root=ROOT)
    effective_rabi_hz = None if args.omega_max_mhz is None else float(args.omega_max_mhz) * 1e6
    include_noise = not bool(args.no_noise)
    durations = (
        [float(value) for value in args.durations]
        if args.durations is not None
        else [float(calibration.target_dimensionless_duration)]
    )
    if args.times_ns is not None:
        durations = [
            calibration.physical_gate_time_to_dimensionless(value * 1e-9, effective_rabi_hz)
            for value in args.times_ns
        ]

    model = build_ma2023_model(
        include_noise=include_noise,
        calibration=calibration,
        effective_rabi_hz=effective_rabi_hz,
    )
    ensemble_models = build_ma2023_quasistatic_ensemble(
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        include_noise=include_noise,
        calibration=calibration,
        effective_rabi_hz=effective_rabi_hz,
    )
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=args.num_tslots,
            evo_time=durations[0],
            max_iter=args.max_iter,
            num_restarts=args.num_restarts,
            seed=args.seed,
            init_pulse_type=args.init_pulse_type,
            init_control_scale=args.init_control_scale,
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
            fidelity_target=args.fidelity_target,
            objective_metric=args.objective_metric,
            benchmark_active_channel=True,
            show_progress=args.show_progress,
        ),
        ensemble_models=ensemble_models,
    )

    initial_ctrl_x = None
    initial_ctrl_y = None
    if args.initial_pulse_from_data:
        initial_ctrl_x, initial_ctrl_y, _data_duration = ma2023_fig3_controls(
            num_tslots=args.num_tslots,
            root=ROOT,
        )

    scan, raw_results = optimizer.scan_durations(
        durations,
        initial_ctrl_x=initial_ctrl_x,
        initial_ctrl_y=initial_ctrl_y,
        initial_theta=np.pi / 2.0,
    )
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT))
    results: list[dict[str, object]] = []
    result_files: list[str] = []

    for duration, result in zip(durations, raw_results):
        result_payload = result.to_json()
        duration_label = f"{duration:.3f}".replace(".", "p")
        result_path = output_dir / f"{args.output_prefix}_TOmega_{duration_label}.json"
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        result_files.append(result_path.name)
        diagnostic = average_sink_populations(optimizer, result.ctrl_x, result.ctrl_y)
        summary = summarize_ma2023_result(
            result=result,
            gate_time_dimensionless=duration,
            calibration=calibration,
            effective_rabi_hz=effective_rabi_hz,
        )
        summary.update(diagnostic)
        summary["result_file"] = result_path.name
        results.append(summary)

    best_index = int(np.argmax([float(item["objective_fidelity"]) for item in results]))
    summary_payload = {
        "source": "Ma et al., Nature 622, 279-284 (2023), Fig. 3 reproduction track",
        "calibration": calibration.summary(effective_rabi_hz),
        "include_noise": bool(include_noise),
        "ensemble_size": int(args.ensemble_size),
        "ensemble_seed": int(args.seed),
        "objective_metric": args.objective_metric,
        "durations_dimensionless": durations,
        "scan": scan.to_json(),
        "best_index": int(best_index),
        "best_result_file": result_files[best_index],
        "best_objective_fidelity": float(results[best_index]["objective_fidelity"]),
        "best_gate_time_dimensionless": float(results[best_index]["gate_time_dimensionless"]),
        "best_gate_time_ns": float(results[best_index]["gate_time_ns"]),
        "target_two_qubit_fidelity": float(calibration.target_two_qubit_fidelity),
        "results": results,
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
