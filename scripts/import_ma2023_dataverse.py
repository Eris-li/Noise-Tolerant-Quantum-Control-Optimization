from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "ma2023"
PROCESSED_DIR = RAW_DIR / "processed"


def numeric_series(rows: list[dict[str, str]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(name, "").strip()
        if value == "":
            continue
        values.append(float(value))
    return values


def paired_series(rows: list[dict[str, str]], x_name: str, y_name: str, yerr_name: str | None = None) -> dict[str, list[float]]:
    payload: dict[str, list[float]] = {"x": [], "y": []}
    if yerr_name is not None:
        payload["yerr"] = []
    for row in rows:
        x_value = row.get(x_name, "").strip()
        y_value = row.get(y_name, "").strip()
        if x_value == "" or y_value == "":
            continue
        payload["x"].append(float(x_value))
        payload["y"].append(float(y_value))
        if yerr_name is not None:
            yerr_value = row.get(yerr_name, "").strip()
            payload["yerr"].append(float(yerr_value) if yerr_value else float("nan"))
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_fig3_payload() -> dict[str, object]:
    fig3 = read_csv(RAW_DIR / "Fig3.csv")
    fig_s2 = read_csv(RAW_DIR / "FigS2.csv")
    fig_s3 = read_csv(RAW_DIR / "FigS3v2.csv")

    time_us = numeric_series(fig3, "Fig3a_Omega_x")
    omega_mhz = numeric_series(fig3, "Fig3a_Omega_y")
    phase_rad = numeric_series(fig3, "Fig3a_phi_y")
    if not (len(time_us) == len(omega_mhz) == len(phase_rad)):
        raise ValueError("Fig3 pulse columns have inconsistent lengths")

    peak_rabi_mhz = float(max(omega_mhz))
    duration_us = float(max(time_us))
    amplitude_fraction = [float(value / peak_rabi_mhz) for value in omega_mhz]
    ctrl_x = [float(a * np.cos(phi)) for a, phi in zip(amplitude_fraction, phase_rad)]
    ctrl_y = [float(a * np.sin(phi)) for a, phi in zip(amplitude_fraction, phase_rad)]
    target_dimensionless_duration = float(2.0 * np.pi * peak_rabi_mhz * duration_us)

    true_gate_errors = numeric_series(fig_s3, "FigS3b_TrueGateError")
    extracted_gate_errors = numeric_series(fig_s3, "FigS3b_ExtractedGateError")
    extracted_gate_error_err = numeric_series(fig_s3, "FigS3b_ExtractedGateError_err")

    return {
        "source": {
            "doi": "10.7910/DVN/TJ6OIF",
            "article": "https://www.nature.com/articles/s41586-023-06438-1",
            "raw_files": ["Fig3.csv", "FigS2.csv", "FigS3v2.csv"],
        },
        "calibration": {
            "fig3_duration_us": duration_us,
            "fig3_peak_rabi_mhz": peak_rabi_mhz,
            "fig3_peak_rabi_hz": float(peak_rabi_mhz * 1e6),
            "target_dimensionless_duration": target_dimensionless_duration,
            "target_two_qubit_fidelity": 0.980,
            "target_two_qubit_error": 0.020,
            "nearby_rydberg_detuning_over_omega": 5.8,
        },
        "pulse": {
            "time_us": time_us,
            "omega_mhz": omega_mhz,
            "phase_rad": phase_rad,
            "amplitude_fraction": amplitude_fraction,
            "ctrl_x_fraction": ctrl_x,
            "ctrl_y_fraction": ctrl_y,
        },
        "fig3": {
            "bloch_01": {
                "x": numeric_series(fig3, "Fig3b_|01>_x"),
                "y": numeric_series(fig3, "Fig3b_|01>_y"),
                "z": numeric_series(fig3, "Fig3b_|01>_z"),
            },
            "bloch_11": {
                "x": numeric_series(fig3, "Fig3b_|11>_x"),
                "y": numeric_series(fig3, "Fig3b_|11>_y"),
                "z": numeric_series(fig3, "Fig3b_|11>_z"),
            },
            "bell_signal": {
                "p00": paired_series(fig3, "Fig3e_P00_x", "Fig3e_P00_y", "Fig3e_P00_yerr"),
                "single_points": {
                    "p00": numeric_series(fig3, "Fig3d_|00>_y"),
                    "p11": numeric_series(fig3, "Fig3d_|11>_y"),
                    "f_spam_half": numeric_series(fig3, "Fig3d_Fsp/2"),
                },
            },
            "raw_survival": paired_series(fig3, "Fig3f_P|00>_x", "Fig3f_P|00>_y", "Fig3f_P|00>_yerr"),
        },
        "fig_s2": {
            "target_intensity": paired_series(fig_s2, "FigS2b_TargetIntensity_x", "FigS2b_TargetIntensity_y"),
            "real_intensity": paired_series(
                fig_s2,
                "FigS2b_RealIntensity_x",
                "FigS2b_RealIntensity_y",
                "FigS2b_RealIntensity_yerr",
            ),
            "target_phase": paired_series(fig_s2, "FigS2c_TargetPhase_x", "FigS2c_TargetPhase_y"),
            "real_phase": paired_series(
                fig_s2,
                "FigS2c_RealPhase_x",
                "FigS2c_RealPhase_y",
                "FigS2c_RealPhase_yerr",
            ),
        },
        "fig_s3": {
            "true_gate_error": true_gate_errors,
            "extracted_gate_error": extracted_gate_errors,
            "extracted_gate_error_err": extracted_gate_error_err,
            "mean_true_gate_error": float(np.mean(true_gate_errors)),
            "mean_extracted_gate_error": float(np.mean(extracted_gate_errors)),
        },
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fig3_payload = build_fig3_payload()
    (PROCESSED_DIR / "fig3_time_optimal_gate.json").write_text(
        json.dumps(fig3_payload, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "processed_files": ["fig3_time_optimal_gate.json"],
        "ignored_files": sorted(path.name for path in RAW_DIR.glob("*Zone.Identifier")),
    }
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"processed_dir": str(PROCESSED_DIR), **manifest}, indent=2))


if __name__ == "__main__":
    main()
