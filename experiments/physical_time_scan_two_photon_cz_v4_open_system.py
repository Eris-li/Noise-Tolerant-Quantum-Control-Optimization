from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    summarize_yb171_v4_result,
    yb171_v4_default_omega_max_hz,
    yb171_experimental_calibration,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan physical gate times for the v4 open-system model while optimizing "
            "the lower-leg amplitude sequence Omega(t) and phase sequence phi(t) "
            "under a fixed experimental Omega_max."
        )
    )
    parser.add_argument(
        "--times-ns",
        nargs="+",
        type=float,
        default=[100.0, 120.0, 136.0, 160.0, 200.0],
        help="Physical gate durations in ns.",
    )
    parser.add_argument(
        "--omega-max-mhz",
        type=float,
        default=None,
        help="Maximum effective Rabi rate in MHz. Defaults to the calibrated experimental limit.",
    )
    parser.add_argument("--num-tslots", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--init-pulse-type", type=str, default="SINE")
    parser.add_argument("--init-control-scale", type=float, default=0.75)
    parser.add_argument("--fidelity-target", type=float, default=0.999)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="two_photon_cz_v4_open_system_physical_time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    calibration = yb171_experimental_calibration()
    omega_max_hz = (
        yb171_v4_default_omega_max_hz()
        if args.omega_max_mhz is None
        else float(args.omega_max_mhz) * 1e6
    )
    model = build_yb171_v4_calibrated_model(
        include_noise=True,
        effective_rabi_hz=omega_max_hz,
    )
    ensemble_models = build_yb171_v4_quasistatic_ensemble(
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        include_noise=True,
        effective_rabi_hz=omega_max_hz,
    )
    durations_ns = [float(value) for value in args.times_ns]
    durations_dimless = [
        calibration.physical_gate_time_to_dimensionless(value * 1e-9, effective_rabi_hz=omega_max_hz)
        for value in durations_ns
    ]

    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=args.num_tslots,
            evo_time=durations_dimless[0],
            max_iter=args.max_iter,
            num_restarts=args.num_restarts,
            seed=args.seed,
            init_pulse_type=args.init_pulse_type,
            init_control_scale=args.init_control_scale,
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
            fidelity_target=args.fidelity_target,
            show_progress=True,
        ),
        ensemble_models=ensemble_models,
    )

    scan, raw_results = optimizer.scan_durations(
        durations_dimless,
        initial_theta=np.pi / 2.0,
    )
    raw_scan_payload = {
        "gate_times_ns": durations_ns,
        "durations_dimensionless": durations_dimless,
        "omega_max_hz": omega_max_hz,
        "omega_max_mhz": omega_max_hz / 1e6,
        "ensemble_size": int(args.ensemble_size),
        "ensemble_seed": int(args.seed),
        "scan_result": scan.to_json(),
    }
    (artifacts / f"{args.output_prefix}_raw_scan.json").write_text(
        json.dumps(raw_scan_payload, indent=2),
        encoding="utf-8",
    )

    result_summaries: list[dict[str, object]] = []
    for gate_time_ns, result in zip(durations_ns, raw_results):
        summary = summarize_yb171_v4_result(
            result=result,
            gate_time_ns=gate_time_ns,
            omega_max_hz=omega_max_hz,
            model=model,
        )
        result_summaries.append(summary)
        (artifacts / f"{args.output_prefix}_{gate_time_ns:.1f}ns.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    best_index = int(np.argmax([summary["probe_fidelity"] for summary in result_summaries]))
    summary_payload = {
        "calibration": calibration.summary(effective_rabi_hz=omega_max_hz),
        "omega_max_hz": float(omega_max_hz),
        "omega_max_mhz": float(omega_max_hz / 1e6),
        "ensemble_size": int(args.ensemble_size),
        "ensemble_seed": int(args.seed),
        "gate_times_ns": durations_ns,
        "dimensionless_gate_times": durations_dimless,
        "best_gate_time_ns": float(durations_ns[best_index]),
        "best_probe_fidelity": float(result_summaries[best_index]["probe_fidelity"]),
        "best_channel_fidelity": float(result_summaries[best_index]["channel_fidelity"]),
        "fidelity_target": float(args.fidelity_target),
        "target_reached": bool(scan.target_reached),
        "results": result_summaries,
    }
    (artifacts / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
