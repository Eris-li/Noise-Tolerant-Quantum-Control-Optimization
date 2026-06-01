from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neutral_yb.optimization.grape import ClosedSystemGRAPE
from neutral_yb.optimization.shelved_cr_phase_grape import (
    ShelvedCRPhaseGRAPEConfig,
    unwrap_for_plot,
)


def default_dense_time_grids_ns() -> dict[float, list[float]]:
    return {
        5.0: [240, 260, 280, 300, 320, 340, 360, 380],
        10.0: [120, 135, 150, 160, 170, 180, 190, 210, 240],
        20.0: [60, 75, 90, 105, 120, 135, 150, 160, 170, 180, 190],
    }


@dataclass(frozen=True)
class UVDenseEdgeScanConfig:
    output_dir: Path
    time_grids_ns: dict[float, list[float]] = field(default_factory=default_dense_time_grids_ns)
    edge_values_ns: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0, 40.0, 80.0])
    threshold: float = 0.999
    num_tslots: int = 64
    max_iter: int = 360
    smoothness_weight: float = 1e-4
    curvature_weight: float = 1e-4
    rydberg_lifetime_s: float = 65e-6
    blockade_shift_mhz: float = 160.0

    @property
    def scan_json_path(self) -> Path:
        return self.output_dir / "rydberg_decay_65us_dense_time_scan_results.json"

    @property
    def summary_json_path(self) -> Path:
        return self.output_dir / "rydberg_decay_65us_dense_min_times.json"

    @property
    def selected_phase_json_path(self) -> Path:
        return self.output_dir / "rydberg_decay_65us_dense_selected_phase_rows.json"

    @property
    def scan_csv_path(self) -> Path:
        return self.output_dir / "rydberg_decay_65us_dense_time_scan_summary.csv"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "rydberg_decay_65us_dense_min_times.csv"


def dense_time_is_allowed(total_time_ns: float, edge_ns: float) -> bool:
    return float(total_time_ns) + 1e-9 >= 2.0 * float(edge_ns) + 20.0


def config_metadata(config: UVDenseEdgeScanConfig) -> dict[str, object]:
    return {
        "threshold": float(config.threshold),
        "noise_model": "Rydberg decay as non-Hermitian no-jump term",
        "fidelity_definition": "no-jump Kraus process fidelity |Tr(D_CZ^dagger K_comp)|^2 / 16",
        "rydberg_lifetime_s": float(config.rydberg_lifetime_s),
        "uses_basis_functions": False,
        "uses_regularization": True,
        "regularization_model": "wrapped first- and second-difference phase penalty",
        "smoothness_weight": float(config.smoothness_weight),
        "curvature_weight": float(config.curvature_weight),
        "num_tslots": int(config.num_tslots),
        "max_iter": int(config.max_iter),
        "blockade_shift_mhz": float(config.blockade_shift_mhz),
    }


def _initial_starts(
    grape: ClosedSystemGRAPE,
    previous_variables: np.ndarray | None,
    omega_max_mhz: float,
    edge_ns: float,
    time_index: int,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if previous_variables is not None and previous_variables.size == grape.num_tslots + 2:
        starts.append(np.asarray(previous_variables, dtype=np.float64))

    seed_base = int(1000 * omega_max_mhz + 10 * edge_ns + time_index)
    starts.append(grape.initial_guess(71000 + seed_base))
    starts.append(grape.initial_guess(93000 + seed_base))
    return starts


def run_uv_edge_scan(config: UVDenseEdgeScanConfig) -> list[dict[str, object]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for omega, time_grid in sorted(config.time_grids_ns.items()):
        for edge in config.edge_values_ns:
            previous_variables: np.ndarray | None = None
            for time_index, total_time in enumerate(sorted(float(value) for value in time_grid)):
                if not dense_time_is_allowed(total_time, edge):
                    continue

                grape = ClosedSystemGRAPE.shelved_cr_phase(
                    ShelvedCRPhaseGRAPEConfig(
                        omega_max_mhz=float(omega),
                        total_time_ns=float(total_time),
                        edge_time_ns=float(edge),
                        num_tslots=config.num_tslots,
                        blockade_shift_mhz=config.blockade_shift_mhz,
                        smoothness_weight=config.smoothness_weight,
                        curvature_weight=config.curvature_weight,
                        rydberg_lifetime_s=config.rydberg_lifetime_s,
                    ),
                    include_rydberg_decay=True,
                )
                result, evaluated = grape.optimize(
                    _initial_starts(grape, previous_variables, omega, edge, time_index),
                    max_iter=config.max_iter,
                )
                previous_variables = np.asarray(result.x, dtype=np.float64).copy()

                row = {
                    "omega_max_mhz": float(omega),
                    "edge_ns": float(edge),
                    "total_time_ns": float(total_time),
                    "num_tslots": int(config.num_tslots),
                    "max_iter": int(config.max_iter),
                    "smoothness_weight": float(config.smoothness_weight),
                    "curvature_weight": float(config.curvature_weight),
                    "rydberg_lifetime_s": float(config.rydberg_lifetime_s),
                    "blockade_shift_mhz": float(config.blockade_shift_mhz),
                    "fidelity": float(evaluated["fidelity"]),
                    "process_fidelity": float(evaluated["process_fidelity"]),
                    "no_jump_average_fidelity": float(evaluated["no_jump_average_fidelity"]),
                    "active_population": float(evaluated["active_population"]),
                    "loss_proxy": float(evaluated["loss_proxy"]),
                    "passed": bool(float(evaluated["fidelity"]) >= config.threshold),
                    "success": bool(result.success),
                    "num_iter": int(result.nit),
                    "alpha": float(evaluated["alpha"]),
                    "beta": float(evaluated["beta"]),
                    "phases": [float(value) for value in evaluated["phases"]],
                }
                rows.append(row)
                print(
                    f"Omega/2pi={omega:4.0f} MHz edge={edge:5.0f} ns T={total_time:6.0f} ns "
                    f"F={row['fidelity']:.9f} success={row['success']}",
                    flush=True,
                )

    print(f"UV edge scan completed: {len(rows)} points in {time.perf_counter() - started:.1f} s")
    return rows


def summarize_uv_edge_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for omega in sorted({float(row["omega_max_mhz"]) for row in rows}):
        for edge in sorted({float(row["edge_ns"]) for row in rows if float(row["omega_max_mhz"]) == omega}):
            group = sorted(
                [row for row in rows if float(row["omega_max_mhz"]) == omega and float(row["edge_ns"]) == edge],
                key=lambda row: float(row["total_time_ns"]),
            )
            if not group:
                continue
            passed = [row for row in group if bool(row["passed"])]
            best = max(group, key=lambda row: float(row["fidelity"]))
            selected = dict(passed[0] if passed else best)
            selected["selection_kind"] = "first above threshold" if passed else "best below threshold"
            selected["shortest_passing_time_ns"] = None if not passed else float(passed[0]["total_time_ns"])
            selected["best_time_ns"] = float(best["total_time_ns"])
            selected["best_fidelity"] = float(best["fidelity"])
            selected_rows.append(selected)
            summary.append(
                {
                    "omega_max_mhz": float(omega),
                    "edge_ns": float(edge),
                    "shortest_passing_time_ns": None if not passed else float(passed[0]["total_time_ns"]),
                    "first_passing_fidelity": None if not passed else float(passed[0]["fidelity"]),
                    "best_time_ns": float(best["total_time_ns"]),
                    "best_fidelity": float(best["fidelity"]),
                    "scanned_times_ns": [float(row["total_time_ns"]) for row in group],
                }
            )
    return summary, selected_rows


def write_uv_edge_artifacts(
    config: UVDenseEdgeScanConfig,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = config_metadata(config)
    config.scan_json_path.write_text(json.dumps({**metadata, "results": rows}, indent=2), encoding="utf-8")
    config.summary_json_path.write_text(json.dumps({**metadata, "summary": summary}, indent=2), encoding="utf-8")
    config.selected_phase_json_path.write_text(
        json.dumps({**metadata, "rows": selected_rows}, indent=2),
        encoding="utf-8",
    )

    result_columns = [
        "omega_max_mhz",
        "edge_ns",
        "total_time_ns",
        "num_tslots",
        "max_iter",
        "smoothness_weight",
        "curvature_weight",
        "fidelity",
        "process_fidelity",
        "no_jump_average_fidelity",
        "active_population",
        "loss_proxy",
        "passed",
        "success",
        "num_iter",
        "alpha",
        "beta",
    ]
    with config.scan_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in result_columns} for row in rows])

    summary_columns = [
        "omega_max_mhz",
        "edge_ns",
        "shortest_passing_time_ns",
        "first_passing_fidelity",
        "best_time_ns",
        "best_fidelity",
        "scanned_times_ns",
    ]
    with config.summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in summary_columns} for row in summary])


def load_uv_edge_artifacts(config: UVDenseEdgeScanConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows = json.loads(config.scan_json_path.read_text(encoding="utf-8"))["results"]
    summary = json.loads(config.summary_json_path.read_text(encoding="utf-8"))["summary"]
    selected_rows = json.loads(config.selected_phase_json_path.read_text(encoding="utf-8"))["rows"]
    return rows, summary, selected_rows


def _edge_blues(edge_values: list[float]) -> dict[float, tuple[float, float, float, float]]:
    unique_edges = sorted({float(value) for value in edge_values})
    color_values = np.linspace(0.35, 0.90, len(unique_edges))
    return {edge: plt.cm.Blues(value) for edge, value in zip(unique_edges, color_values)}


def _omega_reds(omega_values: list[float]) -> dict[float, tuple[float, float, float, float]]:
    unique_omegas = sorted({float(value) for value in omega_values})
    color_values = np.linspace(0.35, 0.90, len(unique_omegas))
    return {omega: plt.cm.Reds(value) for omega, value in zip(unique_omegas, color_values)}


def plot_uv_edge_artifacts(
    config: UVDenseEdgeScanConfig,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_fidelity_curves(config, rows)
    _plot_best_fidelity_vs_edge(config, summary)
    _plot_selected_time_vs_edge(config, summary)
    _plot_selected_phase_traces(config, selected_rows)


def _plot_fidelity_curves(config: UVDenseEdgeScanConfig, rows: list[dict[str, object]]) -> None:
    omega_values = sorted({float(row["omega_max_mhz"]) for row in rows})
    figure, axes = plt.subplots(len(omega_values), 1, figsize=(9.5, 8.8), sharex=False)
    axes_array = np.atleast_1d(axes)
    for axis, omega in zip(axes_array, omega_values):
        omega_rows = [row for row in rows if float(row["omega_max_mhz"]) == omega]
        colors = _edge_blues([float(row["edge_ns"]) for row in omega_rows])
        for edge in sorted(colors):
            edge_rows = sorted(
                [row for row in omega_rows if float(row["edge_ns"]) == edge],
                key=lambda row: float(row["total_time_ns"]),
            )
            axis.plot(
                [float(row["total_time_ns"]) for row in edge_rows],
                [float(row["fidelity"]) for row in edge_rows],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=colors[edge],
                label=f"edge {edge:g} ns",
            )
        axis.axhline(config.threshold, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(rf"Noisy dense time scan, $\Omega_{{\max}}/2\pi={omega:g}$ MHz")
        axis.set_ylabel("No-jump process fidelity")
        axis.grid(True, alpha=0.30)
        axis.legend(ncol=3, fontsize=8)
    axes_array[-1].set_xlabel("UV segment time T (ns)")
    figure.tight_layout()
    figure.savefig(config.output_dir / "rydberg_decay_65us_dense_fidelity_curves.png", dpi=180)
    plt.close(figure)


def _plot_best_fidelity_vs_edge(config: UVDenseEdgeScanConfig, summary: list[dict[str, object]]) -> None:
    omega_values = sorted({float(row["omega_max_mhz"]) for row in summary})
    colors = _omega_reds(omega_values)
    figure, axis = plt.subplots(figsize=(9.8, 5.2))
    for omega in omega_values:
        omega_rows = sorted(
            [row for row in summary if float(row["omega_max_mhz"]) == omega],
            key=lambda row: float(row["edge_ns"]),
        )
        axis.plot(
            [float(row["edge_ns"]) for row in omega_rows],
            [float(row["best_fidelity"]) for row in omega_rows],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=colors[omega],
            label=f"{omega:g} MHz",
        )
    axis.axhline(config.threshold, color="black", linestyle="--", linewidth=1.0, label=f"threshold {config.threshold:g}")
    axis.set_xlabel("Gaussian edge length per side (ns)")
    axis.set_ylabel("Best no-jump process fidelity")
    axis.set_title("Noisy dense scan: best process fidelity vs Gaussian edge length")
    axis.grid(True, alpha=0.30)
    axis.legend(title=r"$\Omega_{\max}/2\pi$")
    figure.tight_layout()
    figure.savefig(config.output_dir / "rydberg_decay_65us_dense_best_fidelity_vs_edge.png", dpi=180)
    plt.close(figure)


def _plot_selected_time_vs_edge(config: UVDenseEdgeScanConfig, summary: list[dict[str, object]]) -> None:
    omega_values = sorted({float(row["omega_max_mhz"]) for row in summary})
    colors = _omega_reds(omega_values)
    figure, axis = plt.subplots(figsize=(9.8, 5.2))
    for omega in omega_values:
        omega_rows = sorted(
            [row for row in summary if float(row["omega_max_mhz"]) == omega],
            key=lambda row: float(row["edge_ns"]),
        )
        selected_times = [
            float(row["shortest_passing_time_ns"])
            if row["shortest_passing_time_ns"] is not None
            else float(row["best_time_ns"])
            for row in omega_rows
        ]
        axis.plot(
            [float(row["edge_ns"]) for row in omega_rows],
            selected_times,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=colors[omega],
            label=f"{omega:g} MHz",
        )
        for row, selected_time in zip(omega_rows, selected_times):
            if row["shortest_passing_time_ns"] is None:
                axis.scatter([float(row["edge_ns"])], [selected_time], marker="x", s=70, color=colors[omega])
    axis.set_xlabel("Gaussian edge length per side (ns)")
    axis.set_ylabel("Selected UV segment time T (ns)")
    axis.set_title("Noisy dense scan: selected time vs Gaussian edge length")
    axis.grid(True, alpha=0.30)
    axis.legend(title=r"$\Omega_{\max}/2\pi$")
    figure.tight_layout()
    figure.savefig(config.output_dir / "rydberg_decay_65us_dense_selected_time_vs_edge.png", dpi=180)
    plt.close(figure)


def _plot_selected_phase_traces(config: UVDenseEdgeScanConfig, selected_rows: list[dict[str, object]]) -> None:
    omega_values = sorted({float(row["omega_max_mhz"]) for row in selected_rows})
    figure, axes = plt.subplots(len(omega_values), 1, figsize=(8.8, 9.0), sharex=False)
    axes_array = np.atleast_1d(axes)
    for axis, omega in zip(axes_array, omega_values):
        omega_rows = sorted(
            [row for row in selected_rows if float(row["omega_max_mhz"]) == omega],
            key=lambda row: float(row["edge_ns"]),
        )
        colors = _edge_blues([float(row["edge_ns"]) for row in omega_rows])
        for row in omega_rows:
            phases = [float(value) for value in row["phases"]]
            times = np.linspace(0.0, float(row["total_time_ns"]), len(phases))
            edge = float(row["edge_ns"])
            label = f"edge {edge:g} ns, T {float(row['total_time_ns']):.0f} ns"
            if row.get("selection_kind") == "best below threshold":
                label += ", best below threshold"
            axis.plot(times, unwrap_for_plot(phases), linewidth=1.8, color=colors[edge], label=label)
        axis.set_title(rf"Selected dense-scan phase traces, $\Omega_{{\max}}/2\pi={omega:g}$ MHz")
        axis.set_ylabel("Unwrapped phase (rad)")
        axis.grid(True, alpha=0.30)
        axis.legend(fontsize=8)
    axes_array[-1].set_xlabel("Time in UV segment (ns)")
    figure.tight_layout()
    figure.savefig(config.output_dir / "rydberg_decay_65us_dense_selected_phase_traces.png", dpi=180)
    plt.close(figure)
