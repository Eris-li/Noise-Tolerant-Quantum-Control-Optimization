from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutral_yb.analysis.uv_edge_scan import (
    UVDenseEdgeScanConfig,
    load_uv_edge_artifacts,
    plot_uv_edge_artifacts,
    run_uv_edge_scan,
    summarize_uv_edge_rows,
    write_uv_edge_artifacts,
)
from neutral_yb.config.artifact_paths import ensure_artifact_dir, yb171_uv_edge_artifacts_dir


def build_config(args: argparse.Namespace) -> UVDenseEdgeScanConfig:
    if args.smoke:
        return UVDenseEdgeScanConfig(
            output_dir=ensure_artifact_dir(yb171_uv_edge_artifacts_dir(ROOT) / "smoke"),
            time_grids_ns={10.0: [60.0, 80.0]},
            edge_values_ns=[0.0, 10.0],
            num_tslots=8,
            max_iter=2,
            threshold=0.95,
        )

    return UVDenseEdgeScanConfig(
        output_dir=ensure_artifact_dir(Path(args.output_dir) if args.output_dir else yb171_uv_edge_artifacts_dir(ROOT)),
        num_tslots=args.num_tslots,
        max_iter=args.max_iter,
        threshold=args.threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the 171Yb UV rising/falling-edge scan from reusable src APIs."
    )
    parser.add_argument("--output-dir", default=None, help="Artifact directory. Defaults to artifacts/v5/closed_cr_edge_time_optimal_scan.")
    parser.add_argument("--num-tslots", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=360)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--recompute", action="store_true", help="Run the expensive GRAPE scan instead of loading existing JSON.")
    parser.add_argument("--replot", action="store_true", help="Regenerate figures from JSON artifacts.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny two-point scan for fast API verification.")
    args = parser.parse_args()

    config = build_config(args)
    artifacts_exist = (
        config.scan_json_path.exists()
        and config.summary_json_path.exists()
        and config.selected_phase_json_path.exists()
    )

    if args.recompute or args.smoke or not artifacts_exist:
        rows = run_uv_edge_scan(config)
        summary, selected_rows = summarize_uv_edge_rows(rows)
        write_uv_edge_artifacts(config, rows, summary, selected_rows)
        plot_uv_edge_artifacts(config, rows, summary, selected_rows)
    else:
        rows, summary, selected_rows = load_uv_edge_artifacts(config)
        if args.replot:
            plot_uv_edge_artifacts(config, rows, summary, selected_rows)

    print(
        f"UV edge scan ready: {len(rows)} points, "
        f"{len(summary)} edge/Omega summaries, artifacts={config.output_dir}"
    )


if __name__ == "__main__":
    main()
