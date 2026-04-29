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
from neutral_yb.config.ma2023_calibration import (  # noqa: E402
    build_ma2023_six_level_model,
    load_ma2023_fig3_data,
    ma2023_experimental_calibration,
)
from neutral_yb.models.ma2023_pulse import Ma2023GaussianEdgePulse  # noqa: E402
from neutral_yb.optimization.ma2023_six_level_grape import (  # noqa: E402
    Ma2023SixLevelChebyshevPhaseRateOptimizer,
    Ma2023SixLevelGRAPEConfig,
    Ma2023SixLevelPhaseOptimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Method-first Ma 2023 reproduction using the Methods six-level perfect-blockade model."
    )
    parser.add_argument("--num-tslots", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gaussian-edge-fraction", type=float, default=0.20)
    parser.add_argument("--gaussian-edge-sigma-fraction", type=float, default=0.08)
    parser.add_argument("--phase-smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--phase-curvature-weight", type=float, default=1e-3)
    parser.add_argument("--phase-parameterization", choices=("chebyshev", "direct"), default="chebyshev")
    parser.add_argument("--chebyshev-degree", type=int, default=13)
    parser.add_argument("--chebyshev-init-scale", type=float, default=1.0)
    parser.add_argument("--chebyshev-coefficient-bound", type=float, default=20.0)
    parser.add_argument("--optimize-phase-origin", action="store_true", default=False)
    parser.add_argument(
        "--initial-direct-result",
        type=Path,
        default=None,
        help="Optional direct-phase GRAPE result JSON to project into Chebyshev phase-rate coefficients.",
    )
    parser.add_argument(
        "--initial-chebyshev-result",
        type=Path,
        default=None,
        help="Optional Chebyshev result JSON to continue optimizing from its coefficients.",
    )
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output", type=str, default="ma2023_six_level_method.json")
    return parser.parse_args()


def rydberg_transition_ranges(optimizer: Ma2023SixLevelPhaseOptimizer, ctrl_x: np.ndarray, ctrl_y: np.ndarray) -> dict[str, object]:
    ranges: dict[str, object] = {}
    for sector in optimizer.model.sector_labels():
        final_state = optimizer.evolve_basis(ctrl_x, ctrl_y, optimizer.model.transition_subspace_indices(sector)[0])
        indices = optimizer.model.transition_subspace_indices(sector)
        populations = np.abs(final_state[list(indices)]) ** 2
        ranges[sector] = {
            "final_computational_population": float(populations[0]),
            "final_rydberg_population": float(np.sum(populations[1:])),
        }
    return ranges


def main() -> None:
    args = parse_args()
    calibration = ma2023_experimental_calibration(root=ROOT)
    fig3_data = load_ma2023_fig3_data(ROOT)
    model = build_ma2023_six_level_model(
        include_noise=not bool(args.no_noise),
        calibration=calibration,
    )
    envelope = Ma2023GaussianEdgePulse(
        num_tslots=args.num_tslots,
        edge_fraction=args.gaussian_edge_fraction,
        sigma_fraction=args.gaussian_edge_sigma_fraction,
    ).envelope()
    optimizer_class = (
        Ma2023SixLevelChebyshevPhaseRateOptimizer
        if args.phase_parameterization == "chebyshev"
        else Ma2023SixLevelPhaseOptimizer
    )
    optimizer = optimizer_class(
        model=model,
        config=Ma2023SixLevelGRAPEConfig(
            num_tslots=args.num_tslots,
            evo_time=float(calibration.target_dimensionless_duration),
            max_iter=args.max_iter,
            num_restarts=args.num_restarts,
            seed=args.seed,
            phase_smoothness_weight=args.phase_smoothness_weight,
            phase_curvature_weight=args.phase_curvature_weight,
            chebyshev_degree=args.chebyshev_degree,
            chebyshev_init_scale=args.chebyshev_init_scale,
            chebyshev_coefficient_bound=args.chebyshev_coefficient_bound,
            optimize_phase_origin=bool(args.optimize_phase_origin),
            show_progress=args.show_progress,
        ),
        envelope=envelope,
    )
    initial_variables = None
    if args.initial_direct_result is not None and args.initial_chebyshev_result is not None:
        raise ValueError("Use only one of --initial-direct-result or --initial-chebyshev-result")
    if args.initial_direct_result is not None:
        if not isinstance(optimizer, Ma2023SixLevelChebyshevPhaseRateOptimizer):
            raise ValueError("--initial-direct-result requires --phase-parameterization chebyshev")
        direct_payload = json.loads(args.initial_direct_result.read_text(encoding="utf-8"))
        initial_variables = optimizer.variables_from_slot_phases(
            np.asarray(direct_payload["optimized_phase_rad_bounded"], dtype=np.float64),
            theta0=float(direct_payload.get("theta0", 0.0)),
            theta1=float(direct_payload.get("theta1", 0.0)),
        )
    if args.initial_chebyshev_result is not None:
        if not isinstance(optimizer, Ma2023SixLevelChebyshevPhaseRateOptimizer):
            raise ValueError("--initial-chebyshev-result requires --phase-parameterization chebyshev")
        cheb_payload = json.loads(args.initial_chebyshev_result.read_text(encoding="utf-8"))
        initial_variables = np.concatenate(
            [
                np.asarray(cheb_payload["phase_rate_chebyshev_coefficients"], dtype=np.float64),
                np.array(
                    [
                        float(cheb_payload.get("phase_origin", 0.0)),
                        float(cheb_payload.get("theta0", 0.0)),
                        float(cheb_payload.get("theta1", 0.0)),
                    ]
                    if args.optimize_phase_origin
                    else [
                        float(cheb_payload.get("theta0", 0.0)),
                        float(cheb_payload.get("theta1", 0.0)),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
    result = optimizer.optimize(initial_variables=initial_variables) if initial_variables is not None else optimizer.optimize()
    payload = result.to_json()
    payload.update(
        {
            "reproduction_mode": "method_first_six_level",
            "dataverse_pulse_used_as_initial_condition": False,
            "dataverse_used_only_for_calibration_and_comparison": True,
            "source": fig3_data["source"],
            "model": {
                "name": "Ma2023PerfectBlockadeSixLevelModel",
                "phase_parameterization": args.phase_parameterization,
                "optimize_phase_origin": bool(args.optimize_phase_origin),
                "basis": list(model.basis_labels()),
                "delta_r_over_omega": float(model.delta_r),
                "delta_m_over_omega": float(model.delta_m),
                "include_loss_state": bool(model.include_loss_state),
                "noise": {
                    "rydberg_decay_rate": float(model.noise.rydberg_decay_rate),
                    "rydberg_dephasing_rate": float(model.noise.rydberg_dephasing_rate),
                    "common_detuning": float(model.noise.common_detuning),
                    "rabi_amplitude_scale": float(model.noise.rabi_amplitude_scale),
                },
            },
            "calibration": calibration.summary(),
            "fixed_amplitude_envelope": [float(value) for value in envelope],
            "transition_population_summary": rydberg_transition_ranges(optimizer, result.ctrl_x, result.ctrl_y),
            "initial_direct_result": None
            if args.initial_direct_result is None
            else str(args.initial_direct_result),
            "initial_chebyshev_result": None
            if args.initial_chebyshev_result is None
            else str(args.initial_chebyshev_result),
        }
    )
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT) / "from_method")
    output_path = output_dir / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **{k: payload[k] for k in ("fidelity", "process_fidelity", "leakage")}}, indent=2))


if __name__ == "__main__":
    main()
