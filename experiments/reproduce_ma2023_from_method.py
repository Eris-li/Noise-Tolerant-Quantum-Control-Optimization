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
    summarize_ma2023_result,
)
from neutral_yb.models.ma2023_pulse import (  # noqa: E402
    Ma2023GaussianEdgePulse,
    controls_from_envelope_phase,
    phase_regularization,
    validate_phase_only_pulse,
    wrap_phase,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Method-first reproduction of Ma et al. Nature 2023 Fig. 3. "
            "This does not use the Dataverse pulse as an optimizer initial condition."
        )
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=None,
        help="Dimensionless T * Omega_max values. Defaults to the Dataverse Fig. 3 duration.",
    )
    parser.add_argument("--num-tslots", type=int, default=96)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--init-pulse-type", type=str, default="SINE")
    parser.add_argument("--init-control-scale", type=float, default=0.75)
    parser.add_argument("--objective-metric", type=str, default="special_state")
    parser.add_argument("--fidelity-target", type=float, default=0.98)
    parser.add_argument("--phase-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian-edge-fraction", type=float, default=0.20)
    parser.add_argument("--gaussian-edge-sigma-fraction", type=float, default=0.08)
    parser.add_argument("--phase-smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--phase-curvature-weight", type=float, default=1e-3)
    parser.add_argument("--radial-amplitude-bound-weight", type=float, default=1000.0)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="ma2023_from_method")
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


def active_bloch_trajectory(
    optimizer: OpenSystemGRAPEOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    state_label: str,
) -> dict[str, list[float]]:
    model = optimizer.model
    if state_label == "01":
        rho0 = qutip.ket2dm(model.probe_kets(theta=0.0)[0][0])
    elif state_label == "11":
        rho0 = qutip.ket2dm(model.probe_kets(theta=0.0)[1][0])
    else:
        raise ValueError(f"Unsupported state label: {state_label}")

    times, states = optimizer.trajectory(ctrl_x, ctrl_y, rho0)
    left, right = model.active_gate_indices()
    bloch_x: list[float] = []
    bloch_y: list[float] = []
    bloch_z: list[float] = []
    for state in states:
        matrix = np.asarray(state.full(), dtype=np.complex128)
        rho_active = matrix[np.ix_([left, right], [left, right])]
        trace = np.trace(rho_active)
        if abs(trace) > 1e-12:
            rho_active = rho_active / trace
        bloch_x.append(float(2.0 * np.real(rho_active[0, 1])))
        bloch_y.append(float(2.0 * np.imag(rho_active[1, 0])))
        bloch_z.append(float(np.real(rho_active[0, 0] - rho_active[1, 1])))
    return {
        "time_dimensionless": [float(value) for value in times],
        "x": bloch_x,
        "y": bloch_y,
        "z": bloch_z,
    }


def transition_bloch_trajectory(
    optimizer: OpenSystemGRAPEOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    state_label: str,
) -> dict[str, list[float]]:
    model = optimizer.model
    if state_label == "01":
        rho0 = qutip.ket2dm(model.probe_kets(theta=0.0)[0][0])
        subspace = [0, 1]
    elif state_label == "11":
        rho0 = qutip.ket2dm(model.probe_kets(theta=0.0)[1][0])
        subspace = [2, 3]
    else:
        raise ValueError(f"Unsupported state label: {state_label}")

    times, states = optimizer.trajectory(ctrl_x, ctrl_y, rho0)
    bloch_x: list[float] = []
    bloch_y: list[float] = []
    bloch_z: list[float] = []
    subspace_population: list[float] = []
    for state in states:
        matrix = np.asarray(state.full(), dtype=np.complex128)
        rho_sub = matrix[np.ix_(subspace, subspace)]
        trace = np.real(np.trace(rho_sub))
        subspace_population.append(float(trace))
        if abs(trace) > 1e-12:
            rho_sub = rho_sub / trace
        bloch_x.append(float(2.0 * np.real(rho_sub[0, 1])))
        bloch_y.append(float(2.0 * np.imag(rho_sub[1, 0])))
        bloch_z.append(float(np.real(rho_sub[0, 0] - rho_sub[1, 1])))
    return {
        "time_dimensionless": [float(value) for value in times],
        "x": bloch_x,
        "y": bloch_y,
        "z": bloch_z,
        "subspace_population": subspace_population,
    }


def optimize_phase_only(
    optimizer: OpenSystemGRAPEOptimizer,
    *,
    envelope: np.ndarray,
    max_iter: int,
    num_restarts: int,
    seed: int,
    phase_smoothness_weight: float,
    phase_curvature_weight: float,
    initial_theta: float,
    show_progress: bool,
):
    rng = np.random.default_rng(seed)
    slots = int(envelope.size)
    best = None

    def variables_to_controls(variables: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        phases = wrap_phase(np.asarray(variables[:-1], dtype=np.float64))
        theta = float(variables[-1])
        ctrl_x, ctrl_y = controls_from_envelope_phase(envelope, phases)
        return ctrl_x, ctrl_y, theta, phases

    for restart in range(int(num_restarts)):
        grid = np.linspace(0.0, 2.0 * np.pi, slots, endpoint=False, dtype=np.float64)
        if restart == 0:
            phases0 = 0.25 * np.sin(grid)
        else:
            phases0 = rng.uniform(-np.pi, np.pi, size=slots)
        variables0 = np.concatenate([phases0, np.array([initial_theta], dtype=np.float64)])

        def objective_and_gradient(variables: np.ndarray) -> tuple[float, np.ndarray]:
            ctrl_x, ctrl_y, theta, phases = variables_to_controls(variables)
            cartesian_variables = np.concatenate([ctrl_x, ctrl_y, np.array([theta], dtype=np.float64)])
            objective, cartesian_gradient = optimizer.objective_and_gradient(cartesian_variables)
            grad_x = cartesian_gradient[:slots]
            grad_y = cartesian_gradient[slots : 2 * slots]
            grad_theta = cartesian_gradient[-1]
            phase_cost, phase_grad = phase_regularization(
                phases,
                phase_smoothness_weight,
                phase_curvature_weight,
            )
            gradient_phases = (
                grad_x * (-envelope * np.sin(phases))
                + grad_y * (envelope * np.cos(phases))
                + phase_grad
            )
            gradient = np.concatenate([gradient_phases, np.array([grad_theta], dtype=np.float64)])
            return float(objective + phase_cost), gradient

        result = minimize(
            fun=objective_and_gradient,
            x0=variables0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * slots + [(0.0, 2.0 * np.pi)],
            options={"maxiter": int(max_iter)},
        )
        ctrl_x, ctrl_y, theta, _phases = variables_to_controls(result.x)
        candidate = optimizer._result_from_variables(
            np.concatenate([ctrl_x, ctrl_y, np.array([theta], dtype=np.float64)]),
            num_iter=int(result.nit),
            num_fid_func_calls=int(result.nfev),
            wall_time=0.0,
            termination_reason=str(result.message),
            success=bool(result.success),
        )
        if show_progress:
            print(
                f"[phase-only] restart {restart + 1}/{num_restarts} "
                f"F={candidate.objective_fidelity:.6f} max_amp={np.max(candidate.amplitudes):.6f}",
                flush=True,
            )
        if best is None or candidate.objective_fidelity > best.objective_fidelity:
            best = candidate

    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    fig3_data = load_ma2023_fig3_data(ROOT)
    calibration = ma2023_experimental_calibration(root=ROOT)
    include_noise = not bool(args.no_noise)
    durations = (
        [float(value) for value in args.durations]
        if args.durations is not None
        else [float(calibration.target_dimensionless_duration)]
    )
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT) / "from_method")

    model = build_ma2023_model(include_noise=include_noise, calibration=calibration)
    ensemble_models = build_ma2023_quasistatic_ensemble(
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        include_noise=include_noise,
        calibration=calibration,
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
            radial_amplitude_bound_weight=args.radial_amplitude_bound_weight,
            fidelity_target=args.fidelity_target,
            objective_metric=args.objective_metric,
            benchmark_active_channel=True,
            show_progress=args.show_progress,
        ),
        ensemble_models=ensemble_models,
    )

    if args.phase_only and len(durations) != 1:
        raise ValueError("Phase-only method-first mode currently supports exactly one duration")

    if args.phase_only:
        envelope = Ma2023GaussianEdgePulse(
            num_tslots=args.num_tslots,
            edge_fraction=args.gaussian_edge_fraction,
            sigma_fraction=args.gaussian_edge_sigma_fraction,
        ).envelope()
        raw_results = [
            optimize_phase_only(
                optimizer,
                envelope=envelope,
                max_iter=args.max_iter,
                num_restarts=args.num_restarts,
                seed=args.seed,
                phase_smoothness_weight=args.phase_smoothness_weight,
                phase_curvature_weight=args.phase_curvature_weight,
                initial_theta=np.pi / 2.0,
                show_progress=args.show_progress,
            )
        ]
        scan = {
            "durations": durations,
            "fidelities": [float(raw_results[0].objective_fidelity)],
            "best_duration": durations[0] if raw_results[0].objective_fidelity >= args.fidelity_target else None,
            "best_fidelity": float(raw_results[0].objective_fidelity),
            "target_reached": bool(raw_results[0].objective_fidelity >= args.fidelity_target),
        }
    else:
        scan, raw_results = optimizer.scan_durations(durations, initial_theta=np.pi / 2.0)
        scan = scan.to_json()
    result_files: list[str] = []
    summaries: list[dict[str, object]] = []
    for duration, result in zip(durations, raw_results):
        duration_label = f"{duration:.3f}".replace(".", "p")
        result_payload = result.to_json()
        result_payload["method_first"] = True
        result_payload["dataverse_pulse_used_as_initial_condition"] = False
        result_payload["phase_parameterization"] = "phase_only_fixed_gaussian_edges" if args.phase_only else "cartesian"
        if args.phase_only:
            result_payload["fixed_amplitude_envelope"] = [float(value) for value in envelope]
            bounded_phase = wrap_phase(np.arctan2(result.ctrl_y, result.ctrl_x))
            result_payload["optimized_phase_rad_bounded"] = [float(value) for value in bounded_phase]
            result_payload["pulse_validation"] = validate_phase_only_pulse(
                result.amplitudes,
                bounded_phase,
            )
        result_payload["bloch_01"] = transition_bloch_trajectory(optimizer, result.ctrl_x, result.ctrl_y, "01")
        result_payload["bloch_11"] = transition_bloch_trajectory(optimizer, result.ctrl_x, result.ctrl_y, "11")
        result_payload["computational_active_bloch_01"] = active_bloch_trajectory(
            optimizer,
            result.ctrl_x,
            result.ctrl_y,
            "01",
        )
        result_payload["computational_active_bloch_11"] = active_bloch_trajectory(
            optimizer,
            result.ctrl_x,
            result.ctrl_y,
            "11",
        )
        result_path = output_dir / f"{args.output_prefix}_TOmega_{duration_label}.json"
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        result_files.append(result_path.name)

        summary = summarize_ma2023_result(
            result=result,
            gate_time_dimensionless=duration,
            calibration=calibration,
        )
        summary.update(average_sink_populations(optimizer, result.ctrl_x, result.ctrl_y))
        summary["result_file"] = result_path.name
        summaries.append(summary)

    best_index = int(np.argmax([float(item["objective_fidelity"]) for item in summaries]))
    summary_payload = {
        "reproduction_mode": "method_first",
        "dataverse_pulse_used_as_initial_condition": False,
        "dataverse_used_only_for_calibration_and_comparison": True,
        "source": fig3_data["source"],
        "calibration": calibration.summary(),
        "include_noise": bool(include_noise),
        "ensemble_size": int(args.ensemble_size),
        "ensemble_seed": int(args.seed),
        "objective_metric": args.objective_metric,
        "num_tslots": int(args.num_tslots),
        "max_iter": int(args.max_iter),
        "num_restarts": int(args.num_restarts),
        "init_pulse_type": args.init_pulse_type,
        "init_control_scale": float(args.init_control_scale),
        "phase_only": bool(args.phase_only),
        "gaussian_edge_fraction": float(args.gaussian_edge_fraction),
        "gaussian_edge_sigma_fraction": float(args.gaussian_edge_sigma_fraction),
        "phase_smoothness_weight": float(args.phase_smoothness_weight),
        "phase_curvature_weight": float(args.phase_curvature_weight),
        "radial_amplitude_bound_weight": float(args.radial_amplitude_bound_weight),
        "durations_dimensionless": durations,
        "scan": scan,
        "best_index": int(best_index),
        "best_result_file": result_files[best_index],
        "best_objective_fidelity": float(summaries[best_index]["objective_fidelity"]),
        "best_gate_time_dimensionless": float(summaries[best_index]["gate_time_dimensionless"]),
        "best_gate_time_ns": float(summaries[best_index]["gate_time_ns"]),
        "target_two_qubit_fidelity": float(calibration.target_two_qubit_fidelity),
        "results": summaries,
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
