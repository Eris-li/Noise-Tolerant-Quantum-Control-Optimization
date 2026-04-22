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

from neutral_yb.config.artifact_paths import ensure_artifact_dir, v1_artifacts_dir
from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
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


def run_scan(
    durations: list[float],
    num_tslots: int,
    max_iter: int,
) -> tuple[object, list[object]]:
    model = GlobalCZ4DModel(species=idealised_yb171())
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=num_tslots,
            evo_time=durations[0],
            max_iter=max_iter,
            phase_seed=11,
            init_phase_spread=0.8,
            fidelity_target=0.9999,
        ),
    )
    return optimizer.scan_durations(durations)


def fit_time_optimal_from_fine_scan(durations: list[float], fidelities: list[float]) -> dict[str, float]:
    times = np.asarray(durations, dtype=float)
    errors = 1.0 - np.asarray(fidelities, dtype=float)
    mask = times <= 7.7
    t_fit = times[mask]
    e_fit = errors[mask]

    def model(t: np.ndarray, t_star: float, scale: float) -> np.ndarray:
        return scale * np.maximum(t_star - t, 0.0) ** 2

    popt, _ = curve_fit(
        model,
        t_fit,
        e_fit,
        p0=[7.61, 0.05],
        bounds=([7.5, 0.0], [7.8, 10.0]),
    )
    return {
        "t_star": float(popt[0]),
        "scale": float(popt[1]),
    }


def main() -> None:
    coarse_durations = list(reversed(frange(1.0, 10.0, 0.5)))
    coarse_scan, coarse_results = run_scan(coarse_durations, num_tslots=100, max_iter=250)

    artifacts = ensure_artifact_dir(v1_artifacts_dir(ROOT))

    model = GlobalCZ4DModel(species=idealised_yb171())
    writer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(num_tslots=100, evo_time=10.0, max_iter=250),
    )
    writer.save_scan(coarse_scan, artifacts / "freeze_v1_global_cz_coarse_scan.json")

    threshold = 0.9999
    qualifying = [res.evo_time for res in coarse_results if res.fidelity >= threshold]
    if not qualifying:
        raise RuntimeError("No coarse-scan point reached fidelity >= 0.9999")

    coarse_lower = min(qualifying)
    fine_start = max(1.0, coarse_lower - 0.5)
    fine_stop = min(10.0, coarse_lower + 0.5)
    fine_durations = list(reversed(frange(fine_start, fine_stop, 0.025)))
    fine_scan, fine_results = run_scan(fine_durations, num_tslots=100, max_iter=300)
    writer.save_scan(fine_scan, artifacts / "freeze_v1_global_cz_fine_scan.json")

    fine_qualifying = [res for res in fine_results if res.fidelity >= threshold]
    if not fine_qualifying:
        raise RuntimeError("No fine-scan point reached fidelity >= 0.9999")

    optimal = min(fine_qualifying, key=lambda res: res.evo_time)
    writer.save_result(optimal, artifacts / "freeze_v1_global_cz_optimal.json")

    fit = fit_time_optimal_from_fine_scan(fine_scan.durations, fine_scan.fidelities)
    (artifacts / "freeze_v1_global_cz_fit.json").write_text(
        json.dumps(fit, indent=2),
        encoding="utf-8",
    )

    print("Freeze candidate scan completed")
    print(f"Coarse threshold reach starts at T*Omega_max = {coarse_lower}")
    print(f"Fine optimal T*Omega_max = {optimal.evo_time}")
    print(f"Fine optimal fidelity = {optimal.fidelity}")
    print(f"Fitted T*Omega_max = {fit['t_star']}")
    print(f"Saved coarse scan, fine scan, and optimal pulse to {artifacts}")


if __name__ == "__main__":
    main()
