from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ma2023NoiseTrace:
    """One Monte Carlo realization of non-Markovian two-qubit gate noise."""

    common_detuning: np.ndarray
    rabi_scale: np.ndarray
    phase_offset: np.ndarray

    def validate(self, num_tslots: int) -> None:
        expected = (int(num_tslots),)
        if self.common_detuning.shape != expected:
            raise ValueError("common_detuning has the wrong length")
        if self.rabi_scale.shape != expected:
            raise ValueError("rabi_scale has the wrong length")
        if self.phase_offset.shape != expected:
            raise ValueError("phase_offset has the wrong length")


@dataclass(frozen=True)
class Ma2023NoiseTraceConfig:
    """Parameters for trace-based non-Markovian noise simulation.

    All frequencies are dimensionless in units of the reference angular Rabi
    frequency used by the Hamiltonian. The default values are intentionally
    zero except for the number of slots; measured spectra should be supplied
    explicitly when available.
    """

    num_tslots: int
    quasistatic_detuning_rms: float = 0.0
    intensity_noise_rms_fractional: float = 0.0
    phase_noise_rms_rad: float = 0.0
    seed: int = 0


def doppler_detuning_rms_from_t2_star(
    *,
    t2_star_s: float,
    omega_ref_rad_s: float,
) -> float:
    """Convert Gaussian Ramsey 1/e time to dimensionless detuning RMS.

    For a quasistatic Gaussian detuning with standard deviation sigma_delta,
    Ramsey contrast decays as exp[-(sigma_delta * t)^2 / 2]. A 1/e time gives
    sigma_delta = sqrt(2) / T2*.
    """

    if t2_star_s <= 0.0:
        raise ValueError("t2_star_s must be positive")
    if omega_ref_rad_s <= 0.0:
        raise ValueError("omega_ref_rad_s must be positive")
    return float(np.sqrt(2.0) / (omega_ref_rad_s * t2_star_s))


def generate_noise_trace(config: Ma2023NoiseTraceConfig) -> Ma2023NoiseTrace:
    rng = np.random.default_rng(config.seed)
    slots = int(config.num_tslots)
    common_detuning_value = rng.normal(0.0, float(config.quasistatic_detuning_rms))
    common_detuning = np.full(slots, common_detuning_value, dtype=np.float64)
    rabi_scale = 1.0 + rng.normal(0.0, float(config.intensity_noise_rms_fractional), size=slots)
    rabi_scale = np.clip(rabi_scale, 0.0, None).astype(np.float64)
    phase_offset = rng.normal(0.0, float(config.phase_noise_rms_rad), size=slots).astype(np.float64)
    trace = Ma2023NoiseTrace(
        common_detuning=common_detuning,
        rabi_scale=rabi_scale,
        phase_offset=phase_offset,
    )
    trace.validate(slots)
    return trace


def apply_noise_trace_to_controls(
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    trace: Ma2023NoiseTrace,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl_x = np.asarray(ctrl_x, dtype=np.float64)
    ctrl_y = np.asarray(ctrl_y, dtype=np.float64)
    trace.validate(ctrl_x.size)
    amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2) * trace.rabi_scale
    phases = np.arctan2(ctrl_y, ctrl_x) + trace.phase_offset
    return (
        (amplitudes * np.cos(phases)).astype(np.float64),
        (amplitudes * np.sin(phases)).astype(np.float64),
    )
