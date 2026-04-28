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

from neutral_yb.config.artifact_paths import (  # noqa: E402
    ensure_artifact_dir,
    ma2023_time_optimal_2q_dir,
)
from neutral_yb.config.species import idealised_yb171  # noqa: E402
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel  # noqa: E402
from neutral_yb.optimization.global_phase_grape import (  # noqa: E402
    GlobalPhaseOptimizationConfig,
    PaperGlobalPhaseOptimizer,
)


REFERENCE_DURATION = 7.612


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the Ma et al. Nature 2023 time-optimal two-qubit-gate "
            "reproduction from the ideal global-CZ model."
        )
    )
    parser.add_argument("--duration", type=float, default=REFERENCE_DURATION)
    parser.add_argument("--num-tslots", type=int, default=99)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--phase-seed", type=int, default=11)
    parser.add_argument("--init-phase-spread", type=float, default=0.8)
    parser.add_argument("--fidelity-target", type=float, default=0.9999)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def filename_for_run(duration: float, num_tslots: int, max_iter: int, num_restarts: int) -> str:
    duration_label = f"{duration:.3f}".replace(".", "p")
    return (
        "ma2023_time_optimal_2q_gate_"
        f"TOmega_{duration_label}_N{num_tslots}_iter{max_iter}_R{num_restarts}.json"
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_artifact_dir(ma2023_time_optimal_2q_dir(ROOT))
    model = GlobalCZ4DModel(species=idealised_yb171())
    optimizer = PaperGlobalPhaseOptimizer(
        model=model,
        config=GlobalPhaseOptimizationConfig(
            num_tslots=args.num_tslots,
            evo_time=args.duration,
            max_iter=args.max_iter,
            phase_seed=args.phase_seed,
            init_phase_spread=args.init_phase_spread,
            fidelity_target=args.fidelity_target,
            num_restarts=args.num_restarts,
            show_progress=args.show_progress,
        ),
    )

    result = optimizer.optimize()
    result_path = output_dir / filename_for_run(
        args.duration,
        args.num_tslots,
        args.max_iter,
        args.num_restarts,
    )
    optimizer.save_result(result, result_path)

    summary = {
        "source": "Ma et al., Nature 622, 279-284 (2023), Fig. 3",
        "method_reference": "Jandura and Pupillo, Quantum 6, 712 (2022)",
        "target_dimensionless_duration": REFERENCE_DURATION,
        "model": "ideal infinite-blockade global CZ, reduced 4D basis",
        "duration": float(args.duration),
        "num_tslots": int(args.num_tslots),
        "num_restarts": int(args.num_restarts),
        "fidelity": float(result.fidelity),
        "theta": float(result.theta),
        "phase_min": float(np.min(result.phases)),
        "phase_max": float(np.max(result.phases)),
        "result_file": result_path.name,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Ma 2023 time-optimal 2Q reproduction seed completed")
    print(f"T * Omega_max = {args.duration}")
    print(f"Fidelity = {result.fidelity:.12f}")
    print(f"Theta = {result.theta:.12f}")
    print(f"Saved result to {result_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
