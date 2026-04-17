from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy.optimize import curve_fit

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import (
    OpenSystemGRAPEConfig,
    OpenSystemGRAPEOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def build_model() -> TwoPhotonCZOpen10DModel:
    return TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        lower_rabi=4.0,
        upper_rabi=4.0,
        intermediate_detuning=8.0,
        blockade_shift=10.0,
        two_photon_detuning_01=0.01,
        two_photon_detuning_11=0.01,
        noise=TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.01,
            common_two_photon_detuning=0.004,
            differential_two_photon_detuning=0.003,
            doppler_detuning_01=0.002,
            doppler_detuning_11=0.004,
            lower_amplitude_scale=0.99,
            upper_amplitude_scale=0.99,
            intermediate_decay_rate=0.025,
            rydberg_decay_rate=0.015,
            intermediate_dephasing_rate=0.004,
            rydberg_dephasing_rate=0.01,
            extra_rydberg_leakage_rate=0.003,
            intermediate_branch_to_qubit=0.45,
            rydberg_branch_to_qubit=0.05,
        ),
    )


def fit_time_optimal(durations: list[float], fidelities: list[float]) -> dict[str, float]:
    times = np.asarray(durations, dtype=float)
    errors = 1.0 - np.asarray(fidelities, dtype=float)
    mask = np.asarray(fidelities, dtype=float) >= 0.995
    if int(np.sum(mask)) < 3:
        mask = np.asarray(fidelities, dtype=float) >= 0.99
    t_fit = times[mask]
    e_fit = errors[mask]
    if t_fit.size < 3:
        raise RuntimeError("Not enough high-fidelity fine-scan points for a stable quadratic fit")

    def model(t: np.ndarray, t_star: float, scale: float) -> np.ndarray:
        return scale * np.maximum(t_star - t, 0.0) ** 2

    popt, _ = curve_fit(
        model,
        t_fit,
        e_fit,
        p0=[float(np.min(t_fit)), 0.05],
        bounds=([float(np.min(t_fit)) - 0.2, 0.0], [float(np.max(t_fit)) + 0.2, 10.0]),
    )
    return {"t_star": float(popt[0]), "scale": float(popt[1])}


def main() -> None:
    threshold = 0.999
    coarse_durations = list(reversed(frange(0.5, 10.0, 0.5)))
    coarse_optimizer = OpenSystemGRAPEOptimizer(
        model=build_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=32,
            evo_time=coarse_durations[0],
            max_iter=3,
            num_restarts=2,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
            show_progress=True,
        ),
    )
    coarse_scan, coarse_results = coarse_optimizer.scan_durations(coarse_durations)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    coarse_optimizer.save_scan(coarse_scan, artifacts / "two_photon_cz_v4_open_system_coarse.json")

    coarse_qualifying = [res for res in coarse_results if res.probe_fidelity >= threshold]
    if not coarse_qualifying:
        raise RuntimeError("No coarse-scan point reached fidelity >= 0.999")

    coarse_threshold_point = min(coarse_qualifying, key=lambda res: res.evo_time)
    fine_durations = list(reversed(frange(7.5, coarse_threshold_point.evo_time, 0.025)))

    fine_optimizer = OpenSystemGRAPEOptimizer(
        model=build_model(),
        config=OpenSystemGRAPEConfig(
            num_tslots=100,
            evo_time=fine_durations[0],
            max_iter=5,
            num_restarts=10,
            seed=17,
            init_pulse_type="SINE",
            init_control_scale=0.08,
            control_smoothness_weight=1e-3,
            control_curvature_weight=2e-3,
            fidelity_target=threshold,
            show_progress=True,
        ),
    )
    fine_scan, fine_results = fine_optimizer.scan_durations(
        fine_durations,
        initial_ctrl_x=coarse_threshold_point.ctrl_x,
        initial_ctrl_y=coarse_threshold_point.ctrl_y,
        initial_theta=coarse_threshold_point.optimized_theta,
    )
    fine_optimizer.save_scan(fine_scan, artifacts / "two_photon_cz_v4_open_system_fine.json")

    fine_qualifying = [res for res in fine_results if res.probe_fidelity >= threshold]
    if not fine_qualifying:
        raise RuntimeError("No fine-scan point reached fidelity >= 0.999")
    optimal = min(fine_qualifying, key=lambda res: res.evo_time)
    fine_optimizer.save_result(optimal, artifacts / "two_photon_cz_v4_open_system_optimal.json")

    best = max(fine_results, key=lambda result: result.probe_fidelity)
    fine_optimizer.save_result(best, artifacts / "two_photon_cz_v4_open_system_best.json")

    fit = fit_time_optimal(fine_scan.durations, fine_scan.fidelities)
    (artifacts / "two_photon_cz_v4_open_system_fit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

    print("Two-stage v4 open-system scan completed")
    print(f"Coarse first threshold point = {coarse_threshold_point.evo_time}")
    print(f"Fine optimal T*Omega_max = {optimal.evo_time}")
    print(f"Fine optimal fidelity = {optimal.probe_fidelity}")
    print(f"Fitted T*Omega_max = {fit['t_star']}")


if __name__ == "__main__":
    main()
