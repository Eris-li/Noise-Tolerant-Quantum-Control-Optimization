from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.ma2023_time_optimal_2q import (
    Ma2023NoiseConfig,
    Ma2023TimeOptimal2QModel,
)
from neutral_yb.models.ma2023_six_level import (
    Ma2023PerfectBlockadeSixLevelModel,
    Ma2023SixLevelNoiseConfig,
)
from neutral_yb.optimization.open_system_grape import OpenSystemGRAPEResult


@dataclass(frozen=True)
class Ma2023ExperimentalCalibration:
    """Experimental-scale parameters for Ma et al. Nature 2023 reproduction.

    Public main-text/extended-figure facts used here:
    - the paper reports one- and two-qubit fidelities 0.9990(1) and 0.980(1);
    - Fig. 3 is the time-optimal two-qubit-gate result;
    - Extended Data Fig. 1 states that a neighboring Rydberg-state detuning is
      5.8 times larger than the UV Rabi frequency.

    Parameters not exposed in the main text remain explicit knobs rather than
    hidden constants. They can be replaced once supplementary/dataverse values
    are imported.
    """

    uv_rabi_hz: float = 1.0e6
    uv_rabi_hz_max: float = 1.0e6
    blockade_shift_over_omega: float = 16.0
    nearby_rydberg_detuning_over_omega: float = 5.8
    rydberg_lifetime_s: float = 65e-6
    metastable_lifetime_s: float = 1.0
    rydberg_t2_s: float | None = None
    uv_pulse_area_fractional_rms: float = 0.0
    quasistatic_detuning_rms_hz: float = 0.0
    differential_detuning_rms_hz: float = 0.0
    blockade_shift_jitter_fractional_rms: float = 0.0
    off_resonant_leakage_per_gate: float = 0.0
    leakage_reference_t_omega: float = 7.612
    target_two_qubit_fidelity: float = 0.980
    target_dimensionless_duration: float = 7.612
    profile_name: str = "ma2023_main_text_extfig_seed"

    def resolve_rabi_hz(self, effective_rabi_hz: float | None = None) -> float:
        return float(self.uv_rabi_hz if effective_rabi_hz is None else effective_rabi_hz)

    def omega_ref_rad_s(self, effective_rabi_hz: float | None = None) -> float:
        return 2.0 * math.pi * self.resolve_rabi_hz(effective_rabi_hz)

    def dimensionless_hamiltonian_frequency(
        self,
        hz: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return float(hz / self.resolve_rabi_hz(effective_rabi_hz))

    def dimensionless_rate_from_lifetime(
        self,
        lifetime_s: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return float(1.0 / (self.omega_ref_rad_s(effective_rabi_hz) * lifetime_s))

    def physical_gate_time_to_dimensionless(
        self,
        gate_time_s: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return float(self.omega_ref_rad_s(effective_rabi_hz) * gate_time_s)

    def dimensionless_gate_time_to_seconds(
        self,
        t_omega: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return float(t_omega / self.omega_ref_rad_s(effective_rabi_hz))

    def open_system_noise(
        self,
        effective_rabi_hz: float | None = None,
    ) -> Ma2023NoiseConfig:
        rydberg_dephasing_rate = 0.0
        if self.rydberg_t2_s is not None and self.rydberg_t2_s > 0.0:
            rydberg_dephasing_rate = self.dimensionless_rate_from_lifetime(
                self.rydberg_t2_s,
                effective_rabi_hz,
            )
        return Ma2023NoiseConfig(
            metastable_loss_rate=self.dimensionless_rate_from_lifetime(
                self.metastable_lifetime_s,
                effective_rabi_hz,
            ),
            rydberg_decay_rate=self.dimensionless_rate_from_lifetime(
                self.rydberg_lifetime_s,
                effective_rabi_hz,
            ),
            rydberg_dephasing_rate=rydberg_dephasing_rate,
            off_resonant_leakage_rate=float(
                self.off_resonant_leakage_per_gate / max(self.leakage_reference_t_omega, 1e-12)
            ),
        )

    def summary(self, effective_rabi_hz: float | None = None) -> dict[str, float | str | None]:
        payload = asdict(self)
        payload["effective_rabi_hz"] = self.resolve_rabi_hz(effective_rabi_hz)
        payload["target_gate_time_ns"] = 1e9 * self.dimensionless_gate_time_to_seconds(
            self.target_dimensionless_duration,
            effective_rabi_hz,
        )
        return payload


def ma2023_processed_fig3_path(root: Path | None = None) -> Path:
    base = Path.cwd() if root is None else Path(root)
    return base / "data" / "ma2023" / "processed" / "fig3_time_optimal_gate.json"


def load_ma2023_fig3_data(root: Path | None = None) -> dict[str, object]:
    path = ma2023_processed_fig3_path(root)
    return json.loads(path.read_text(encoding="utf-8"))


def ma2023_experimental_calibration(
    *,
    root: Path | None = None,
    prefer_processed_data: bool = True,
) -> Ma2023ExperimentalCalibration:
    if prefer_processed_data:
        path = ma2023_processed_fig3_path(root)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload["calibration"]
            return Ma2023ExperimentalCalibration(
                uv_rabi_hz=float(calibration["fig3_peak_rabi_hz"]),
                uv_rabi_hz_max=float(calibration["fig3_peak_rabi_hz"]),
                nearby_rydberg_detuning_over_omega=float(
                    calibration["nearby_rydberg_detuning_over_omega"]
                ),
                target_two_qubit_fidelity=float(calibration["target_two_qubit_fidelity"]),
                target_dimensionless_duration=float(calibration["target_dimensionless_duration"]),
                profile_name="ma2023_dataverse_fig3",
            )
    return Ma2023ExperimentalCalibration()


def ma2023_fig3_controls(
    *,
    num_tslots: int | None = None,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    payload = load_ma2023_fig3_data(root)
    pulse = payload["pulse"]
    time_us = np.asarray(pulse["time_us"], dtype=np.float64)
    ctrl_x = np.asarray(pulse["ctrl_x_fraction"], dtype=np.float64)
    ctrl_y = np.asarray(pulse["ctrl_y_fraction"], dtype=np.float64)
    duration = float(payload["calibration"]["target_dimensionless_duration"])
    if num_tslots is None or int(num_tslots) == ctrl_x.size:
        return ctrl_x, ctrl_y, duration
    target_time = np.linspace(float(time_us[0]), float(time_us[-1]), int(num_tslots), endpoint=True)
    return (
        np.interp(target_time, time_us, ctrl_x).astype(np.float64),
        np.interp(target_time, time_us, ctrl_y).astype(np.float64),
        duration,
    )


def build_ma2023_model(
    *,
    include_noise: bool = True,
    calibration: Ma2023ExperimentalCalibration | None = None,
    effective_rabi_hz: float | None = None,
) -> Ma2023TimeOptimal2QModel:
    resolved = ma2023_experimental_calibration() if calibration is None else calibration
    return Ma2023TimeOptimal2QModel(
        species=idealised_yb171(),
        rabi_frequency=1.0,
        blockade_shift=float(resolved.blockade_shift_over_omega),
        noise=resolved.open_system_noise(effective_rabi_hz) if include_noise else Ma2023NoiseConfig(),
    )


def build_ma2023_six_level_model(
    *,
    include_noise: bool = True,
    calibration: Ma2023ExperimentalCalibration | None = None,
    effective_rabi_hz: float | None = None,
) -> Ma2023PerfectBlockadeSixLevelModel:
    resolved = ma2023_experimental_calibration() if calibration is None else calibration
    noise = Ma2023SixLevelNoiseConfig()
    if include_noise:
        noise = Ma2023SixLevelNoiseConfig(
            rydberg_decay_rate=resolved.dimensionless_rate_from_lifetime(
                resolved.rydberg_lifetime_s,
                effective_rabi_hz,
            ),
            rydberg_dephasing_rate=0.0
            if resolved.rydberg_t2_s is None
            else resolved.dimensionless_rate_from_lifetime(resolved.rydberg_t2_s, effective_rabi_hz),
        )
    return Ma2023PerfectBlockadeSixLevelModel(
        delta_r=float(resolved.nearby_rydberg_detuning_over_omega),
        delta_m=0.0,
        rabi_frequency=1.0,
        include_loss_state=True,
        noise=noise,
    )


def build_ma2023_quasistatic_ensemble(
    *,
    ensemble_size: int,
    seed: int,
    include_noise: bool = True,
    calibration: Ma2023ExperimentalCalibration | None = None,
    effective_rabi_hz: float | None = None,
) -> list[Ma2023TimeOptimal2QModel]:
    resolved = ma2023_experimental_calibration() if calibration is None else calibration
    base_noise = resolved.open_system_noise(effective_rabi_hz) if include_noise else Ma2023NoiseConfig()
    rng = np.random.default_rng(seed)
    models: list[Ma2023TimeOptimal2QModel] = []
    size = max(int(ensemble_size), 1)

    for _ in range(size):
        common_detuning = (
            resolved.dimensionless_hamiltonian_frequency(
                rng.normal(0.0, resolved.quasistatic_detuning_rms_hz),
                effective_rabi_hz,
            )
            if include_noise
            else 0.0
        )
        differential_detuning = (
            resolved.dimensionless_hamiltonian_frequency(
                rng.normal(0.0, resolved.differential_detuning_rms_hz),
                effective_rabi_hz,
            )
            if include_noise
            else 0.0
        )
        blockade_offset = (
            rng.normal(0.0, resolved.blockade_shift_jitter_fractional_rms)
            * resolved.blockade_shift_over_omega
            if include_noise
            else 0.0
        )
        rabi_scale = (
            max(0.0, 1.0 + rng.normal(0.0, resolved.uv_pulse_area_fractional_rms))
            if include_noise
            else 1.0
        )
        noise = replace(
            base_noise,
            common_detuning=float(common_detuning),
            differential_detuning=float(differential_detuning),
            blockade_shift_offset=float(blockade_offset),
            rabi_amplitude_scale=float(rabi_scale),
        )
        models.append(
            Ma2023TimeOptimal2QModel(
                species=idealised_yb171(),
                rabi_frequency=1.0,
                blockade_shift=float(resolved.blockade_shift_over_omega),
                noise=noise,
            )
        )
    return models


def summarize_ma2023_result(
    *,
    result: OpenSystemGRAPEResult,
    gate_time_dimensionless: float,
    calibration: Ma2023ExperimentalCalibration,
    effective_rabi_hz: float | None = None,
) -> dict[str, object]:
    gate_time_s = calibration.dimensionless_gate_time_to_seconds(
        gate_time_dimensionless,
        effective_rabi_hz,
    )
    return {
        "gate_time_dimensionless": float(gate_time_dimensionless),
        "gate_time_ns": float(gate_time_s * 1e9),
        "objective_fidelity": float(result.objective_fidelity),
        "phase_gate_fidelity": float(result.probe_fidelity),
        "active_channel_fidelity": None
        if result.active_channel_fidelity is None
        else float(result.active_channel_fidelity),
        "optimized_theta": float(result.optimized_theta),
        "fid_err": float(result.fid_err),
        "max_amplitude": float(np.max(result.amplitudes)) if result.amplitudes.size else 0.0,
        "mean_amplitude": float(np.mean(result.amplitudes)) if result.amplitudes.size else 0.0,
        "num_iter": int(result.num_iter),
        "success": bool(result.success),
        "termination_reason": result.termination_reason,
    }
