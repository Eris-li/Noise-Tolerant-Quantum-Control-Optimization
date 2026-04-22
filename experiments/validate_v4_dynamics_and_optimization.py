from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys

import numpy as np
import qutip

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.yb171_calibration import (
    build_yb171_v4_calibrated_model,
    build_yb171_v4_quasistatic_ensemble,
    summarize_yb171_v4_result,
    yb171_experimental_calibration,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEConfig, OpenSystemGRAPEOptimizer


def build_validation_model(*, include_noise: bool, effective_rabi_hz: float):
    return replace(
        build_yb171_v4_calibrated_model(include_noise=include_noise, effective_rabi_hz=effective_rabi_hz),
        clock_num_steps=4,
    )


def deterministic_controls(num_tslots: int, amp_bound: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, num_tslots, endpoint=False, dtype=np.float64)
    amplitudes = amp_bound * (0.18 + 0.22 * np.sin(np.pi * (grid + 0.1)) ** 2)
    phases = 0.2 + 1.1 * grid
    ctrl_x = amplitudes * np.cos(phases)
    ctrl_y = amplitudes * np.sin(phases)
    return ctrl_x, ctrl_y


def directional_finite_difference(
    optimizer: OpenSystemGRAPEOptimizer,
    variables: np.ndarray,
    direction: np.ndarray,
    step: float = 1e-7,
) -> float:
    shifted_plus = np.array(variables, copy=True) + step * direction
    shifted_minus = np.array(variables, copy=True) - step * direction
    objective_plus, _ = optimizer.objective_and_gradient(shifted_plus)
    objective_minus, _ = optimizer.objective_and_gradient(shifted_minus)
    return float((objective_plus - objective_minus) / (2.0 * step))


def build_active_basis_operators(optimizer: OpenSystemGRAPEOptimizer) -> list[qutip.Qobj]:
    operators: list[qutip.Qobj] = []
    for active_col in optimizer.active_indices:
        for active_row in optimizer.active_indices:
            operator = np.zeros((optimizer.dimension, optimizer.dimension), dtype=np.complex128)
            operator[active_row, active_col] = 1.0
            operators.append(qutip.Qobj(operator))
    return operators


def projected_active_superoperator(
    optimizer: OpenSystemGRAPEOptimizer,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for operator in build_active_basis_operators(optimizer):
        final = optimizer.evolve_density_matrix(ctrl_x, ctrl_y, operator)
        matrix = np.asarray(final.full(), dtype=np.complex128)
        reduced = np.array(
            [
                matrix[row, col]
                for col in optimizer.active_indices
                for row in optimizer.active_indices
            ],
            dtype=np.complex128,
        )
        columns.append(reduced)
    return np.column_stack(columns)


def validate_dynamics_against_mesolve() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    omega_max_hz = calibration.effective_rabi_hz_max
    gate_time_ns = 136.0
    evo_time = calibration.physical_gate_time_to_dimensionless(gate_time_ns * 1e-9, effective_rabi_hz=omega_max_hz)
    model = build_validation_model(include_noise=True, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=model,
        config=OpenSystemGRAPEConfig(
            num_tslots=3,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
    )
    ctrl_x, ctrl_y = deterministic_controls(optimizer.config.num_tslots, optimizer.amp_bound)
    dt = optimizer.config.dt

    h_d = model.drift_hamiltonian()
    h_x, h_y = model.lower_leg_control_hamiltonians()
    h_clock_x, h_clock_y = model.clock_control_hamiltonians()
    c_ops = model.collapse_operators()
    clock_segments = model.clock_segment_controls()

    probe_states = [qutip.ket2dm(model.probe_kets(theta=0.0)[2][0])]

    final_errors: list[float] = []
    trajectory_errors: list[float] = []
    trace_errors: list[float] = []
    min_eigenvalues: list[float] = []

    for rho0 in probe_states:
        exact_times, exact_traj = optimizer.trajectory(ctrl_x, ctrl_y, rho0)
        reference_traj = [rho0]
        current = rho0
        current_time = 0.0
        reference_times = [current_time]
        for x_value, y_value in zip(clock_segments["prefix_x"], clock_segments["prefix_y"]):
            h_k = h_d + float(x_value) * h_clock_x + float(y_value) * h_clock_y
            l_k = qutip.liouvillian(h_k, c_ops)
            rho_vec = qutip.operator_to_vector(current)
            propagated = (float(clock_segments["prefix_dt"]) * l_k).expm() * rho_vec
            current = qutip.vector_to_operator(propagated)
            current_time += float(clock_segments["prefix_dt"])
            reference_traj.append(current)
            reference_times.append(current_time)
        for x_value, y_value in zip(ctrl_x, ctrl_y):
            h_k = h_d + float(x_value) * h_x + float(y_value) * h_y
            l_k = qutip.liouvillian(h_k, c_ops)
            rho_vec = qutip.operator_to_vector(current)
            propagated = (dt * l_k).expm() * rho_vec
            current = qutip.vector_to_operator(propagated)
            current_time += dt
            reference_traj.append(current)
            reference_times.append(current_time)
        for x_value, y_value in zip(clock_segments["suffix_x"], clock_segments["suffix_y"]):
            h_k = h_d + float(x_value) * h_clock_x + float(y_value) * h_clock_y
            l_k = qutip.liouvillian(h_k, c_ops)
            rho_vec = qutip.operator_to_vector(current)
            propagated = (float(clock_segments["suffix_dt"]) * l_k).expm() * rho_vec
            current = qutip.vector_to_operator(propagated)
            current_time += float(clock_segments["suffix_dt"])
            reference_traj.append(current)
            reference_times.append(current_time)

        final_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(exact_traj[-1].full(), dtype=np.complex128)
                    - np.asarray(reference_traj[-1].full(), dtype=np.complex128)
                )
            )
        )
        trajectory_errors.append(
            float(
                max(
                    np.linalg.norm(
                        np.asarray(state_exact.full(), dtype=np.complex128)
                        - np.asarray(state_ref.full(), dtype=np.complex128)
                    )
                    for state_exact, state_ref in zip(exact_traj, reference_traj)
                )
            )
        )
        trace_errors.extend(abs(float(state.tr().real) - 1.0) for state in exact_traj)
        min_eigenvalues.extend(float(np.min(np.linalg.eigvalsh(np.asarray(state.full(), dtype=np.complex128)).real)) for state in exact_traj)

        if not np.allclose(exact_times, np.asarray(reference_times, dtype=np.float64)):
            raise AssertionError("Internal trajectory time grid mismatch")

    return {
        "uv_segment_time_ns": gate_time_ns,
        "total_gate_time_us": float(
            gate_time_ns / 1000.0 + 2.0 * calibration.clock_pi_pulse_duration_s * 1e6
        ),
        "omega_max_mhz": omega_max_hz / 1e6,
        "num_tslots": optimizer.config.num_tslots,
        "max_final_state_frobenius_error": float(max(final_errors)),
        "max_trajectory_state_frobenius_error": float(max(trajectory_errors)),
        "max_trace_error": float(max(trace_errors)),
        "min_state_eigenvalue": float(min(min_eigenvalues)),
        "passed": bool(
            max(final_errors) < 1e-10
            and max(trajectory_errors) < 1e-10
            and max(trace_errors) < 1e-9
            and min(min_eigenvalues) > -1e-8
        ),
    }


def validate_reduced_channel_consistency() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    omega_max_hz = calibration.effective_rabi_hz_max
    gate_time_ns = 136.0
    evo_time = calibration.physical_gate_time_to_dimensionless(gate_time_ns * 1e-9, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_validation_model(include_noise=True, effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=3,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
    )
    ctrl_x, ctrl_y = deterministic_controls(optimizer.config.num_tslots, optimizer.amp_bound)
    batch_super = optimizer.channel_superoperator(ctrl_x, ctrl_y)
    brute_force_super = projected_active_superoperator(optimizer, ctrl_x, ctrl_y)
    max_abs_error = float(np.max(np.abs(batch_super - brute_force_super)))

    return {
        "max_active_superoperator_abs_error": max_abs_error,
        "passed": bool(max_abs_error < 1e-10),
    }


def validate_special_state_phase_gate_metric() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    omega_max_hz = calibration.effective_rabi_hz_max
    gate_time_ns = 136.0
    evo_time = calibration.physical_gate_time_to_dimensionless(gate_time_ns * 1e-9, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_validation_model(include_noise=True, effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=4,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
    )
    ctrl_x, ctrl_y = deterministic_controls(optimizer.config.num_tslots, 0.35 * optimizer.amp_bound)
    theta = 0.73
    final_state = optimizer.final_phase_state(ctrl_x, ctrl_y)
    fidelity_optimizer, theta_grad = optimizer._phase_gate_fidelity_and_theta_gradient(final_state, theta)
    fidelity_model = optimizer.model.phase_gate_fidelity_from_ket(final_state, theta)
    theta_model, fidelity_model_opt = optimizer.model.optimize_theta_for_ket(final_state)
    theta_optimizer, fidelity_optimizer_opt = optimizer.optimize_theta_for_phase_fidelity(ctrl_x, ctrl_y)
    theta_wrapped_error = float(abs(np.angle(np.exp(1j * (theta_model - theta_optimizer)))))

    return {
        "gate_time_ns": gate_time_ns,
        "fidelity_from_optimizer": float(fidelity_optimizer),
        "fidelity_from_model": float(fidelity_model),
        "theta_gradient": float(theta_grad),
        "theta_model_opt": float(theta_model),
        "theta_optimizer_opt": float(theta_optimizer),
        "theta_wrapped_error": theta_wrapped_error,
        "optimized_fidelity_model": float(fidelity_model_opt),
        "optimized_fidelity_optimizer": float(fidelity_optimizer_opt),
        "passed": bool(
            abs(fidelity_optimizer - fidelity_model) < 1e-12
            and theta_wrapped_error < 1e-9
            and abs(fidelity_model_opt - fidelity_optimizer_opt) < 1e-12
        ),
    }


def validate_exact_gradient() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    omega_max_hz = calibration.effective_rabi_hz_max
    evo_time = calibration.physical_gate_time_to_dimensionless(100e-9, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_validation_model(include_noise=True, effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=2,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            init_pulse_type="ZERO",
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
    )
    ctrl_x, ctrl_y = deterministic_controls(optimizer.config.num_tslots, 0.4 * optimizer.amp_bound)
    theta = 0.9
    variables = np.concatenate([ctrl_x, ctrl_y, np.array([theta], dtype=np.float64)])
    objective, gradient = optimizer.objective_and_gradient(variables)
    direction = np.linspace(0.2, 1.1, variables.size, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    finite_difference = directional_finite_difference(optimizer, variables, direction)
    analytic = float(np.dot(gradient, direction))
    abs_error = abs(analytic - finite_difference)
    rel_error = abs_error / max(1e-9, abs(finite_difference))

    return {
        "objective": float(objective),
        "directional_derivative_analytic": analytic,
        "directional_derivative_finite_difference": finite_difference,
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "passed": bool(abs_error < 5e-4 and rel_error < 5e-3),
    }


def validate_optimization_progress() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    omega_max_hz = calibration.effective_rabi_hz_max
    gate_time_ns = 100.0
    evo_time = calibration.physical_gate_time_to_dimensionless(gate_time_ns * 1e-9, effective_rabi_hz=omega_max_hz)
    optimizer = OpenSystemGRAPEOptimizer(
        model=build_validation_model(include_noise=True, effective_rabi_hz=omega_max_hz),
        config=OpenSystemGRAPEConfig(
            num_tslots=3,
            evo_time=evo_time,
            max_iter=1,
            num_restarts=1,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.25,
            control_smoothness_weight=0.0,
            control_curvature_weight=0.0,
        ),
        ensemble_models=build_yb171_v4_quasistatic_ensemble(
            ensemble_size=1,
            seed=17,
            include_noise=True,
            effective_rabi_hz=omega_max_hz,
        ),
    )
    ctrl_x0, ctrl_y0 = optimizer.initial_guess()
    theta0 = np.pi / 2.0
    initial_fidelity = optimizer.phase_gate_fidelity(ctrl_x0, ctrl_y0, theta0)
    initial_objective, _ = optimizer.objective_and_gradient(
        np.concatenate([ctrl_x0, ctrl_y0, np.array([theta0], dtype=np.float64)])
    )
    result = optimizer.optimize(initial_ctrl_x=ctrl_x0, initial_ctrl_y=ctrl_y0, initial_theta=theta0)

    return {
        "gate_time_ns": gate_time_ns,
        "initial_phase_gate_fidelity": float(initial_fidelity),
        "optimized_phase_gate_fidelity": float(result.probe_fidelity),
        "initial_objective": float(initial_objective),
        "optimized_objective": float(result.objective),
        "improvement": float(result.probe_fidelity - initial_fidelity),
        "optimizer_success": bool(result.success),
        "passed": bool(result.probe_fidelity >= initial_fidelity - 1e-8 and result.objective <= initial_objective + 1e-8),
        "optimized_result": summarize_yb171_v4_result(
            result=result,
            gate_time_ns=gate_time_ns,
            omega_max_hz=omega_max_hz,
            model=optimizer.model,
        ),
    }


def validate_physical_sanity() -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    uv_gate_time_ns = 136.0
    uv_gate_time_s = uv_gate_time_ns * 1e-9
    total_gate_time_s = 2.0 * calibration.clock_pi_pulse_duration_s + uv_gate_time_s
    blockade_over_omega = calibration.blockade_shift_hz / calibration.effective_rabi_hz_max
    quasistatic_detuning_over_omega = (
        calibration.resolved_quasistatic_uv_detuning_rms_hz() / calibration.effective_rabi_hz_max
    )
    uv_gate_over_t2 = uv_gate_time_s / calibration.rydberg_t2_star_s
    uv_gate_over_t1 = uv_gate_time_s / calibration.rydberg_lifetime_s
    total_gate_over_clock_lifetime = total_gate_time_s / calibration.clock_state_lifetime_s

    return {
        "uv_gate_time_ns": uv_gate_time_ns,
        "total_gate_time_us": total_gate_time_s * 1e6,
        "omega_max_mhz": calibration.effective_rabi_hz_max / 1e6,
        "blockade_over_omega_max": float(blockade_over_omega),
        "quasistatic_uv_detuning_over_omega_max": float(quasistatic_detuning_over_omega),
        "uv_gate_over_t2_star": float(uv_gate_over_t2),
        "uv_gate_over_rydberg_lifetime": float(uv_gate_over_t1),
        "total_gate_over_clock_lifetime": float(total_gate_over_clock_lifetime),
        "passed": bool(
            blockade_over_omega > 5.0
            and quasistatic_detuning_over_omega < 0.02
            and uv_gate_over_t2 < 0.1
            and uv_gate_over_t1 < 0.01
            and total_gate_over_clock_lifetime < 1e-3
        ),
    }


def main() -> None:
    print("[v4-validate] dynamics vs mesolve", flush=True)
    dynamics = validate_dynamics_against_mesolve()
    print("[v4-validate] reduced channel consistency", flush=True)
    channel = validate_reduced_channel_consistency()
    print("[v4-validate] paper Eq.(7) special-state metric", flush=True)
    special_state_metric = validate_special_state_phase_gate_metric()
    print("[v4-validate] finite-difference gradient", flush=True)
    gradient = validate_exact_gradient()
    print("[v4-validate] optimization progress", flush=True)
    optimization = validate_optimization_progress()
    print("[v4-validate] physical sanity", flush=True)
    physical = validate_physical_sanity()

    summary = {
        "validation_basis": {
            "time_evolution_reference": "qutip_liouvillian_slice_propagation",
            "optimization_reference": "piecewise_constant_grape_with_exact_frechet_gradient",
            "fidelity_metric": "paper_eq7_special_state_phase_gate_fidelity",
            "caveat": (
                "This validation run checks the default paper Eq.(7) special-state "
                "objective on the active {|01>, |11>} manifold. The optimizer also now "
                "supports an optional active-subspace process-fidelity objective; "
                "reduced-channel exactness is validated separately here."
            ),
        },
        "checks": {
            "dynamics_vs_mesolve": dynamics,
            "reduced_channel_consistency_diagnostic": channel,
            "special_state_phase_gate_metric": special_state_metric,
            "gradient_finite_difference": gradient,
            "optimization_progress": optimization,
            "physical_sanity": physical,
        },
        "all_passed": bool(
            dynamics["passed"]
            and channel["passed"]
            and special_state_metric["passed"]
            and gradient["passed"]
            and optimization["passed"]
            and physical["passed"]
        ),
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / "two_photon_cz_v4_pipeline_validation.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Validation written to {destination}")


if __name__ == "__main__":
    main()
