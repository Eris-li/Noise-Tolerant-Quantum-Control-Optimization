from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.config.artifact_paths import ensure_artifact_dir, evered2023_parallel_cz_dir
from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.evered2023_parallel_cz import (
    Evered2023ParallelCZCalibration,
    Evered2023TimeOptimalPulse,
    build_evered2023_ideal_global_cz_model,
    build_evered2023_two_photon_detuning_model,
)
from neutral_yb.optimization.evered2023_parameterized_grape import (
    Evered2023ParameterizedGRAPEConfig,
    Evered2023ParameterizedGRAPEOptimizer,
    Evered2023TwoPhotonDetuningGRAPEOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def result_with_paper_style_benchmark(result, optimizer_class, model) -> dict[str, object]:
    payload = result.to_json()
    config = Evered2023ParameterizedGRAPEConfig(
        num_tslots=int(result.num_tslots),
        max_iter=1,
        num_restarts=1,
    )
    optimizer = optimizer_class(
        model=model,
        omega_t_over_2pi=float(result.omega_t_over_2pi),
        config=config,
    )
    if hasattr(optimizer, "final_state"):
        final_state = optimizer.final_state(result)
    else:
        phases = optimizer.sampled_phases(result)
        final_state = optimizer.slot_optimizer.final_state(phases)
    theta, process_fidelity = model.optimize_theta_for_state(final_state)
    payload["process_fidelity"] = float(process_fidelity)
    if hasattr(model, "phase_gate_average_fidelity"):
        payload["average_gate_fidelity"] = float(model.phase_gate_average_fidelity(final_state, theta))
    payload["theta_for_benchmark"] = float(theta)
    payload["evered2023_exponential_decay_benchmark"] = (
        model.evered2023_exponential_decay_fidelity(final_state, theta).to_json()
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use parameterized GRAPE to recover the Evered 2023 fixed-amplitude CZ pulse."
    )
    parser.add_argument("--start", type=float, default=1.16, help="Start Omega*T/(2*pi)")
    parser.add_argument("--stop", type=float, default=1.26, help="Stop Omega*T/(2*pi)")
    parser.add_argument("--step", type=float, default=0.005, help="Step in Omega*T/(2*pi)")
    parser.add_argument("--num-tslots", type=int, default=160)
    parser.add_argument("--max-iter", type=int, default=260)
    parser.add_argument("--num-restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--target-fidelity", type=float, default=0.99999)
    parser.add_argument(
        "--model",
        choices=("two-photon", "two-photon-detuning", "effective"),
        default="two-photon",
    )
    parser.add_argument(
        "--intermediate-detuning-over-omega",
        type=float,
        default=None,
        help="Override Delta/Omega. Defaults to the paper scale 7.8 GHz / 4.6 MHz.",
    )
    parser.add_argument(
        "--blockade-over-omega",
        type=float,
        default=None,
        help="Override B/Omega. Defaults to 450 MHz / 4.6 MHz.",
    )
    parser.add_argument(
        "--rabi-calibration",
        choices=("balanced", "paper"),
        default="balanced",
        help="Use balanced Omega_b=Omega_r effective calibration or paper 237/303 MHz single-photon Rabi rates.",
    )
    parser.add_argument(
        "--free-static-detuning",
        action="store_true",
        help="Optimize delta0 instead of fixing it to zero.",
    )
    parser.add_argument(
        "--light-shift-resonance",
        action="store_true",
        help="For the detuning-gauge paper-Rabi model, add delta_res=(Omega_r^2-Omega_b^2)/(4 Delta).",
    )
    parser.add_argument(
        "--dressed-basis",
        action="store_true",
        help="For the detuning-gauge paper-Rabi model, score in the leading-order blue-light dressed basis.",
    )
    parser.add_argument(
        "--include-paper-seed",
        action="store_true",
        help="Use the paper Eq. (1) parameters as the first optimizer start before random restarts.",
    )
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    pulse = Evered2023TimeOptimalPulse()
    calibration = Evered2023ParallelCZCalibration()
    if args.model == "effective":
        model = build_evered2023_ideal_global_cz_model(species=idealised_yb171())
        optimizer_class = Evered2023ParameterizedGRAPEOptimizer
    elif args.model == "two-photon-detuning":
        detuning_ratio = (
            calibration.intermediate_detuning_hz / calibration.omega_over_2pi_hz
            if args.intermediate_detuning_over_omega is None
            else float(args.intermediate_detuning_over_omega)
        )
        blockade_ratio = (
            calibration.blockade_shift_hz / calibration.omega_over_2pi_hz
            if args.blockade_over_omega is None
            else float(args.blockade_over_omega)
        )
        blue_rabi = None
        red_rabi = None
        static_resonance_shift = 0.0
        if args.rabi_calibration == "paper":
            blue_rabi = calibration.blue_rabi_hz / calibration.omega_over_2pi_hz
            red_rabi = calibration.red_rabi_hz / calibration.omega_over_2pi_hz
            if args.light_shift_resonance:
                static_resonance_shift = (red_rabi**2 - blue_rabi**2) / (4.0 * detuning_ratio)
        model = build_evered2023_two_photon_detuning_model(
            species=idealised_yb171(),
            intermediate_detuning_over_effective_rabi=detuning_ratio,
            blockade_shift_over_effective_rabi=blockade_ratio,
            blue_rabi_over_effective_rabi=blue_rabi,
            red_rabi_over_effective_rabi=red_rabi,
            static_resonance_shift=static_resonance_shift,
            use_leading_order_dressed_basis=bool(args.dressed_basis),
        )
        optimizer_class = Evered2023TwoPhotonDetuningGRAPEOptimizer
    else:
        detuning_ratio = (
            calibration.intermediate_detuning_hz / calibration.omega_over_2pi_hz
            if args.intermediate_detuning_over_omega is None
            else float(args.intermediate_detuning_over_omega)
        )
        blockade_ratio = (
            calibration.blockade_shift_hz / calibration.omega_over_2pi_hz
            if args.blockade_over_omega is None
            else float(args.blockade_over_omega)
        )
        from neutral_yb.models.evered2023_parallel_cz import build_evered2023_two_photon_ladder_model

        if args.rabi_calibration == "paper":
            lower_rabi = calibration.blue_rabi_hz / calibration.omega_over_2pi_hz
            upper_rabi = calibration.red_rabi_hz / calibration.omega_over_2pi_hz
        else:
            lower_rabi = (2.0 * detuning_ratio) ** 0.5
            upper_rabi = lower_rabi
        model = build_evered2023_two_photon_ladder_model(
            species=idealised_yb171(),
            lower_rabi=lower_rabi,
            upper_rabi=upper_rabi,
            intermediate_detuning=detuning_ratio,
            blockade_shift=blockade_ratio,
        )
        optimizer_class = Evered2023ParameterizedGRAPEOptimizer
    durations = frange(args.start, args.stop, args.step)
    results = []
    for index, duration in enumerate(durations, start=1):
        config = Evered2023ParameterizedGRAPEConfig(
            num_tslots=args.num_tslots,
            max_iter=args.max_iter,
            num_restarts=args.num_restarts,
            seed=args.seed + 1009 * index,
            include_paper_seed=bool(args.include_paper_seed),
            fix_static_detuning=not bool(args.free_static_detuning),
            static_detuning_value=0.0,
            fidelity_target=float(args.target_fidelity),
            show_progress=args.show_progress,
        )
        optimizer = optimizer_class(
            model=model,
            omega_t_over_2pi=duration,
            config=config,
        )
        result = optimizer.optimize()
        results.append(result)
        print(
            f"[scan] {index:02d}/{len(durations):02d} "
            f"OmegaT/2pi={duration:.4f} F={result.fidelity:.10f} "
            f"A/2pi={result.amplitude_phase_modulation / (2.0 * 3.141592653589793):.5f} "
            f"omega={result.phase_rate:.5f} phi0={result.phase_offset:.5f} "
            f"delta0={result.static_detuning:.5f}",
            flush=True,
        )

    qualifying = [result for result in results if result.fidelity >= float(args.target_fidelity)]
    best_threshold = None if not qualifying else min(qualifying, key=lambda result: result.omega_t_over_2pi)
    best_fidelity = max(results, key=lambda result: result.fidelity)

    result_payloads = [result_with_paper_style_benchmark(result, optimizer_class, model) for result in results]
    best_threshold_payload = (
        None
        if best_threshold is None
        else result_with_paper_style_benchmark(best_threshold, optimizer_class, model)
    )
    best_fidelity_payload = result_with_paper_style_benchmark(best_fidelity, optimizer_class, model)

    payload = {
        "line": "evered2023_parallel_cz",
        "purpose": "GRAPE algorithm validation, not paper-result validation",
        "paper": {
            "title": "High-fidelity parallel entangling gates on a neutral-atom quantum computer",
            "doi": "10.1038/s41586-023-06481-y",
            "article": "Nature 622, 268-272 (2023)",
        },
        "paper_reference_pulse": pulse.to_json(),
        "experimental_scale": calibration.to_json(),
        "grape_setup": {
            "phase_family": "phi(t)=A*cos(omega*t-phi0)+delta0*t",
            "optimizer_fidelity_metric": "diagonal_cz_process_fidelity",
            "paper_style_report_metric": "evered2023_exponential_decay_per_cz",
            "physical_control": (
                "delta(t)=-dphi/dt in the two-photon Hamiltonian"
                if args.model == "two-photon-detuning"
                else "phi(t) as the 420-nm blue-leg optical phase in the two-photon Hamiltonian"
                if args.model == "two-photon"
                else "phi(t) as the effective Rydberg coupling phase"
            ),
            "model": args.model,
            "model_dimension": int(model.dimension()),
            "optimized_parameters": (
                ["A", "omega", "phi0", "delta0", "theta"]
                if args.free_static_detuning
                else ["A", "omega", "phi0", "theta"]
            ),
            "fixed_parameters": {} if args.free_static_detuning else {"delta0": 0.0},
            "rabi_calibration": args.rabi_calibration,
            "initialization": "random restarts over broad non-paper parameter ranges",
            "include_paper_seed": bool(args.include_paper_seed),
            "light_shift_resonance": bool(args.light_shift_resonance),
            "dressed_basis": bool(args.dressed_basis),
            "num_tslots": int(args.num_tslots),
            "max_iter": int(args.max_iter),
            "num_restarts": int(args.num_restarts),
            "seed": int(args.seed),
            "target_fidelity": float(args.target_fidelity),
        },
        "hamiltonian_parameters": {
            "blue_rabi": (
                None
                if args.model == "effective"
                else float(model.blue_rabi)
                if args.model == "two-photon-detuning"
                else float(model.lower_rabi)
            ),
            "red_rabi": (
                None
                if args.model == "effective"
                else float(model.red_rabi)
                if args.model == "two-photon-detuning"
                else float(model.upper_rabi)
            ),
            "intermediate_detuning": None if args.model == "effective" else float(model.intermediate_detuning),
            "blockade_shift": None if args.model == "effective" else float(model.blockade_shift),
            "static_resonance_shift": None
            if args.model != "two-photon-detuning"
            else float(getattr(model, "static_resonance_shift", 0.0)),
            "leading_order_blue_dressing_epsilon": None
            if args.model != "two-photon-detuning" or not getattr(model, "use_leading_order_dressed_basis", False)
            else float(model.leading_order_blue_dressing_epsilon()),
        },
        "scan": {
            "durations_omega_t_over_2pi": [float(value) for value in durations],
            "fidelities": [float(result.fidelity) for result in results],
            "target_fidelity": float(args.target_fidelity),
            "best_threshold_result": best_threshold_payload,
            "best_fidelity_result": best_fidelity_payload,
            "results": result_payloads,
        },
        "scope_notes": [
            "This validates whether GRAPE can recover the fixed-amplitude time-optimal CZ pulse family from random starts.",
            "The single-atom dark-state Hamiltonian is implemented separately for two-photon scattering analysis.",
            "Full paper-level agreement still requires finite two-photon ladder parameters, finite blockade, laser noise, Doppler motion, intermediate-state scattering, Rydberg decay, and benchmarking/SPAM layers.",
        ],
    }

    artifacts = ensure_artifact_dir(evered2023_parallel_cz_dir(ROOT))
    output = artifacts / "evered2023_parameterized_grape_scan.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Evered 2023 parameterized GRAPE validation scan completed")
    if best_threshold is not None:
        print(
            "Earliest threshold point: "
            f"OmegaT/2pi={best_threshold.omega_t_over_2pi:.4f}, "
            f"F={best_threshold.fidelity:.12f}"
        )
    print(
        "Best fidelity point: "
        f"OmegaT/2pi={best_fidelity.omega_t_over_2pi:.4f}, "
        f"F={best_fidelity.fidelity:.12f}"
    )
    print(f"Saved result to {output}")


if __name__ == "__main__":
    main()
