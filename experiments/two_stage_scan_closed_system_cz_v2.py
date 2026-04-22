from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy.optimize import curve_fit

from neutral_yb.config.artifact_paths import ensure_artifact_dir, v2_artifacts_dir
from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.finite_blockade_cz_5d import FiniteBlockadeCZ5DModel
from neutral_yb.optimization.global_phase_grape import (
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def build_model() -> FiniteBlockadeCZ5DModel:
    return FiniteBlockadeCZ5DModel(
        species=idealised_yb171(),
        blockade_shift=8.0,
        static_detuning_01=0.015,
        static_detuning_11=0.015,
        rabi_scale=0.985,
    )


def fit_time_optimal(durations: list[float], fidelities: list[float]) -> dict[str, float]:
    times = np.asarray(durations, dtype=float)
    errors = 1.0 - np.asarray(fidelities, dtype=float)
    mask = errors <= 5e-3
    t_fit = times[mask]
    e_fit = errors[mask]

    def model(t: np.ndarray, t_star: float, scale: float) -> np.ndarray:
        return scale * np.maximum(t_star - t, 0.0) ** 2

    popt, _ = curve_fit(
        model,
        t_fit,
        e_fit,
        p0=[7.9, 0.05],
        bounds=([7.0, 0.0], [9.5, 10.0]),
    )
    return {"t_star": float(popt[0]), "scale": float(popt[1])}


def main() -> None:
    threshold = 0.9999
    coarse_durations = list(reversed(frange(1.0, 10.0, 0.5)))
    model = build_model()
    coarse_optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=100,
            evo_time=coarse_durations[0],
            max_iter=250,
            phase_seed=11,
            init_phase_spread=0.8,
            fidelity_target=threshold,
            smoothness_weight=0.002,
            num_restarts=4,
        ),
    )
    coarse_scan, coarse_results = coarse_optimizer.scan_durations(coarse_durations)

    artifacts = ensure_artifact_dir(v2_artifacts_dir(ROOT))
    coarse_optimizer.save_scan(coarse_scan, artifacts / "closed_system_cz_v2_two_stage_coarse.json")

    qualifying = [res.evo_time for res in coarse_results if res.fidelity >= threshold]
    if not qualifying:
        raise RuntimeError("No coarse-scan point reached fidelity >= 0.9999")

    coarse_lower = min(qualifying)
    fine_start = max(1.0, coarse_lower - 0.5)
    fine_stop = min(10.0, coarse_lower + 0.5)
    fine_durations = list(reversed(frange(fine_start, fine_stop, 0.025)))

    fine_optimizer = PaperGlobalPhaseOptimizer(
        model=build_model(),
        config=GlobalPhaseOptimizationConfig(
            num_tslots=100,
            evo_time=fine_durations[0],
            max_iter=300,
            phase_seed=11,
            init_phase_spread=0.8,
            fidelity_target=threshold,
            smoothness_weight=0.002,
            num_restarts=4,
        ),
    )
    fine_scan, fine_results = fine_optimizer.scan_durations(fine_durations)
    fine_optimizer.save_scan(fine_scan, artifacts / "closed_system_cz_v2_two_stage_fine.json")

    fine_qualifying = [res for res in fine_results if res.fidelity >= threshold]
    if not fine_qualifying:
        raise RuntimeError("No fine-scan point reached fidelity >= 0.9999")
    optimal = min(fine_qualifying, key=lambda res: res.evo_time)
    fine_optimizer.save_result(optimal, artifacts / "closed_system_cz_v2_two_stage_optimal.json")

    fit = fit_time_optimal(fine_scan.durations, fine_scan.fidelities)
    (artifacts / "closed_system_cz_v2_two_stage_fit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

    print("Closed-system corrected CZ two-stage scan completed")
    print(f"Coarse threshold reach starts at T*Omega_max = {coarse_lower}")
    print(f"Fine optimal T*Omega_max = {optimal.evo_time}")
    print(f"Fine optimal fidelity = {optimal.fidelity}")
    print(f"Fitted T*Omega_max = {fit['t_star']}")


if __name__ == "__main__":
    main()
