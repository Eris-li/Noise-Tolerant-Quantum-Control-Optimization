from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_prxq_decay import (
    ARTIFACT_DIR,
    evaluate_full_gate_with_decay,
    load_current_candidates,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def assert_close(actual: float, expected: float, *, atol: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: actual={actual:.12g}, expected={expected:.12g}, atol={atol:.3g}")


def validate_artifacts_exist() -> None:
    required = [
        "to_ar_phases.npz",
        "dr_half_phases.npz",
        "adr_half_phases.npz",
        "to_ar_records.json",
        "dr_records.json",
        "adr_records.json",
        "available_summary.json",
        "decay_summary.json",
        "decay_sensitivity_curves.json",
        "decay_sensitivity_curves.png",
    ]
    for filename in required:
        path = ARTIFACT_DIR / filename
        require(path.exists() and path.stat().st_size > 0, f"missing or empty artifact: {path}")


def validate_record_selection() -> None:
    to_ar_payload = load_json(ARTIFACT_DIR / "to_ar_records.json")
    require(float(to_ar_payload["selected"]["to_nominal_cost"]) < 1e-6, "TO selected physical nominal cost is too large")

    for filename, target, selected_key in [
        ("dr_records.json", "DR", "dr_robust_cost"),
        ("adr_records.json", "ADR", "adr_robust_cost"),
    ]:
        payload = load_json(ARTIFACT_DIR / filename)
        records = payload["scan_records"].get(target, [])
        require(records, f"{filename} has no scan records for {target}")
        selected_cost = float(payload["selected"][selected_key])
        record_key = "nominal_cost" if target == "TO" else "robust_cost"
        best = min(records, key=lambda row: float(row[record_key]))
        assert_close(
            selected_cost,
            float(best[record_key]),
            atol=1e-10,
            label=f"{target} selected cost equals best scan record",
        )


def validate_repeated_half_construction() -> None:
    candidates = load_current_candidates()
    for label in ("DR repeated half", "ADR repeated half"):
        phases = np.asarray(candidates[label]["phases"], dtype=np.float64)
        require(phases.size % 2 == 0, f"{label} phase vector is not even length")
        half = phases.size // 2
        np.testing.assert_allclose(phases[:half], phases[half:], atol=1e-14, rtol=0.0)
        require(bool(candidates[label]["use_doppler_reversal"]), f"{label} must use Doppler reversal")


def validate_closed_summary_recomputes() -> None:
    candidates = load_current_candidates()
    summary = load_json(ARTIFACT_DIR / "available_summary.json")
    rows = {row["label"]: row for row in summary["summary_rows"]}
    for label, candidate in candidates.items():
        phases = np.asarray(candidate["phases"], dtype=np.float64)
        total_time = float(candidate["total_time"])
        use_doppler_reversal = bool(candidate["use_doppler_reversal"])
        nominal = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=0.0,
            use_doppler_reversal=use_doppler_reversal,
        )["process_fidelity"]
        amp_min = min(
            evaluate_full_gate_with_decay(
                phases,
                total_time=total_time,
                gamma_over_omega=0.0,
                amplitude_error=value,
                use_doppler_reversal=use_doppler_reversal,
            )["process_fidelity"]
            for value in (-0.05, 0.05)
        )
        detuning_minus = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=0.0,
            detuning_1=-0.05,
            use_doppler_reversal=use_doppler_reversal,
        )["process_fidelity"]
        detuning_plus = evaluate_full_gate_with_decay(
            phases,
            total_time=total_time,
            gamma_over_omega=0.0,
            detuning_1=0.05,
            use_doppler_reversal=use_doppler_reversal,
        )["process_fidelity"]
        row = rows[label]
        assert_close(nominal, float(row["nominal_fidelity"]), atol=3e-8, label=f"{label} nominal fidelity")
        assert_close(amp_min, float(row["min_amp_fidelity_pm_0p05"]), atol=3e-8, label=f"{label} amp fidelity")
        assert_close(detuning_minus, float(row["detuning_fidelity_m0p05"]), atol=3e-8, label=f"{label} detuning -0.05")
        assert_close(detuning_plus, float(row["detuning_fidelity_p0p05"]), atol=3e-8, label=f"{label} detuning +0.05")


def validate_decay_summary_sanity() -> None:
    decay = load_json(ARTIFACT_DIR / "decay_summary.json")
    rows = {row["label"]: row for row in decay["rows"]}
    for label, row in rows.items():
        require(0.0 <= float(row["decay_active_survival"]) <= 1.0 + 1e-12, f"{label} invalid active survival")
        require(float(row["decay_loss_proxy"]) >= -1e-12, f"{label} negative decay loss")
        require(
            float(row["decay_nominal_fidelity"]) <= float(row["closed_nominal_fidelity"]) + 1e-9,
            f"{label} decay nominal fidelity exceeds closed nominal fidelity",
        )

    require(
        float(rows["AR full CZ"]["decay_min_amp_fidelity_pm_0p05"])
        > float(rows["TO optimized"]["decay_min_amp_fidelity_pm_0p05"]),
        "AR should beat TO under amplitude error after decay",
    )
    dr_det_worst = min(
        float(rows["DR repeated half"]["decay_detuning_fidelity_m0p05"]),
        float(rows["DR repeated half"]["decay_detuning_fidelity_p0p05"]),
    )
    adr_det_worst = min(
        float(rows["ADR repeated half"]["decay_detuning_fidelity_m0p05"]),
        float(rows["ADR repeated half"]["decay_detuning_fidelity_p0p05"]),
    )
    to_det_worst = min(
        float(rows["TO optimized"]["decay_detuning_fidelity_m0p05"]),
        float(rows["TO optimized"]["decay_detuning_fidelity_p0p05"]),
    )
    ar_det_worst = min(
        float(rows["AR full CZ"]["decay_detuning_fidelity_m0p05"]),
        float(rows["AR full CZ"]["decay_detuning_fidelity_p0p05"]),
    )
    require(dr_det_worst > to_det_worst and adr_det_worst > ar_det_worst, "DR/ADR detuning robustness ordering failed")


def validate_decay_curves_symmetry_scale() -> None:
    curves = load_json(ARTIFACT_DIR / "decay_sensitivity_curves.json")
    detuning_grid = np.asarray(curves["detuning_grid"], dtype=np.float64)
    minus = int(np.where(np.isclose(detuning_grid, -0.05))[0][0])
    plus = int(np.where(np.isclose(detuning_grid, 0.05))[0][0])
    for label in ("DR repeated half", "ADR repeated half"):
        fidelity = np.asarray(curves["curves"][label]["decay_detuning_fidelity"], dtype=np.float64)
        asymmetry = abs(float(fidelity[plus] - fidelity[minus]))
        require(asymmetry < 1e-4, f"{label} detuning +/-0.05 asymmetry too large: {asymmetry:.3e}")


def main() -> None:
    validate_artifacts_exist()
    validate_record_selection()
    validate_repeated_half_construction()
    validate_closed_summary_recomputes()
    validate_decay_summary_sanity()
    validate_decay_curves_symmetry_scale()
    print("PRXQ reproduction validation passed")


if __name__ == "__main__":
    main()
