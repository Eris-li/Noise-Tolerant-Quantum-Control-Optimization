from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.models.two_photon_cz_open_10d import (
    TwoPhotonCZOpen10DModel,
    TwoPhotonOpenNoiseConfig,
)


@dataclass(frozen=True)
class Yb171ExperimentalCalibration:
    """Best-effort 171Yb gate calibration compatible with the current v4 model.

    The best published 171Yb CZ gates use single-photon excitation from the
    clock state to a Rydberg state, whereas the current repository model is a
    two-photon ladder with an explicit intermediate level. To keep the v4 code
    usable while moving it closer to experiment, this calibration:

    - uses an effective gate scale taken from high-fidelity neutral-atom CZ work
    - matches the Rydberg lifetime, dephasing, blockade, and laser-area errors
      to reported 171Yb quantities
    - suppresses explicit intermediate-state loss channels because that
      intermediate state is only a surrogate for the present model, not a
      literal 171Yb gate level used in the latest experiments
    """

    effective_rabi_hz: float = 4.6e6
    effective_rabi_hz_max: float = 15e6
    intermediate_detuning_hz: float = 7.8e9
    blockade_shift_hz: float = 160e6
    rydberg_lifetime_s: float = 56e-6
    rydberg_t2_star_s: float = 3.4e-6
    uv_pulse_area_fractional_error: float = 0.004
    residual_common_detuning_hz: float = 10e3
    residual_differential_detuning_hz: float = 5e3
    doppler_detuning_01_hz: float = 10e3
    doppler_detuning_11_hz: float = 15e3
    unwanted_mf_error_per_gate: float = 4.8e-4
    leakage_reference_t_omega: float = 8.0

    def resolve_effective_rabi_hz(self, effective_rabi_hz: float | None = None) -> float:
        return float(self.effective_rabi_hz if effective_rabi_hz is None else effective_rabi_hz)

    def omega_ref_rad_s(self, effective_rabi_hz: float | None = None) -> float:
        return 2.0 * math.pi * self.resolve_effective_rabi_hz(effective_rabi_hz)

    def dimensionless_hamiltonian_frequency(
        self,
        hz: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return hz / self.resolve_effective_rabi_hz(effective_rabi_hz)

    def dimensionless_rate_from_lifetime(
        self,
        lifetime_s: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return 1.0 / (self.omega_ref_rad_s(effective_rabi_hz) * lifetime_s)

    def physical_gate_time_to_dimensionless(
        self,
        gate_time_s: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return self.omega_ref_rad_s(effective_rabi_hz) * gate_time_s

    def dimensionless_gate_time_to_seconds(
        self,
        t_omega: float,
        effective_rabi_hz: float | None = None,
    ) -> float:
        return t_omega / self.omega_ref_rad_s(effective_rabi_hz)

    def derived_one_photon_rabi_hz(self, effective_rabi_hz: float | None = None) -> float:
        omega_hz = self.resolve_effective_rabi_hz(effective_rabi_hz)
        return math.sqrt(2.0 * self.intermediate_detuning_hz * omega_hz)

    def derived_one_photon_rabi_dimensionless(self, effective_rabi_hz: float | None = None) -> float:
        return self.dimensionless_hamiltonian_frequency(
            self.derived_one_photon_rabi_hz(effective_rabi_hz),
            effective_rabi_hz,
        )

    def derived_leakage_rate_dimensionless(self) -> float:
        return self.unwanted_mf_error_per_gate / self.leakage_reference_t_omega

    def nominal_two_photon_parameters(
        self,
        effective_rabi_hz: float | None = None,
    ) -> dict[str, float]:
        lower_upper = self.derived_one_photon_rabi_dimensionless(effective_rabi_hz)
        return {
            "lower_rabi": lower_upper,
            "upper_rabi": lower_upper,
            "intermediate_detuning": self.dimensionless_hamiltonian_frequency(
                self.intermediate_detuning_hz,
                effective_rabi_hz,
            ),
            "blockade_shift": self.dimensionless_hamiltonian_frequency(
                self.blockade_shift_hz,
                effective_rabi_hz,
            ),
            "two_photon_detuning_01": 0.0,
            "two_photon_detuning_11": 0.0,
        }

    def open_system_noise(
        self,
        effective_rabi_hz: float | None = None,
    ) -> TwoPhotonOpenNoiseConfig:
        # The explicit intermediate state is only a ladder surrogate. The latest
        # high-fidelity 171Yb gates use a single-photon clock-to-Rydberg drive,
        # so we do not assign a literal spontaneous-scattering rate here.
        return TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.0,
            common_two_photon_detuning=self.dimensionless_hamiltonian_frequency(
                self.residual_common_detuning_hz,
                effective_rabi_hz,
            ),
            differential_two_photon_detuning=self.dimensionless_hamiltonian_frequency(
                self.residual_differential_detuning_hz,
                effective_rabi_hz,
            ),
            doppler_detuning_01=self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_01_hz,
                effective_rabi_hz,
            ),
            doppler_detuning_11=self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_11_hz,
                effective_rabi_hz,
            ),
            blockade_shift_offset=0.0,
            lower_amplitude_scale=1.0 - self.uv_pulse_area_fractional_error,
            upper_amplitude_scale=1.0 - self.uv_pulse_area_fractional_error,
            intermediate_decay_rate=0.0,
            rydberg_decay_rate=self.dimensionless_rate_from_lifetime(
                self.rydberg_lifetime_s,
                effective_rabi_hz,
            ),
            intermediate_dephasing_rate=0.0,
            rydberg_dephasing_rate=self.dimensionless_rate_from_lifetime(
                self.rydberg_t2_star_s,
                effective_rabi_hz,
            ),
            extra_rydberg_leakage_rate=self.derived_leakage_rate_dimensionless(),
            intermediate_branch_to_qubit=1.0,
            rydberg_branch_to_qubit=0.0,
        )

    def summary(self, effective_rabi_hz: float | None = None) -> dict[str, object]:
        omega_hz = self.resolve_effective_rabi_hz(effective_rabi_hz)
        one_photon_rabi_hz = self.derived_one_photon_rabi_hz(omega_hz)
        return {
            **asdict(self),
            "resolved_effective_rabi_hz": omega_hz,
            "omega_ref_rad_s": self.omega_ref_rad_s(omega_hz),
            "derived_one_photon_rabi_hz": one_photon_rabi_hz,
            "derived_one_photon_rabi_over_omega_ref": self.derived_one_photon_rabi_dimensionless(omega_hz),
            "intermediate_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.intermediate_detuning_hz,
                omega_hz,
            ),
            "blockade_shift_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.blockade_shift_hz,
                omega_hz,
            ),
            "rydberg_decay_rate_dimensionless": self.dimensionless_rate_from_lifetime(
                self.rydberg_lifetime_s,
                omega_hz,
            ),
            "rydberg_dephasing_rate_dimensionless": self.dimensionless_rate_from_lifetime(
                self.rydberg_t2_star_s,
                omega_hz,
            ),
            "residual_common_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.residual_common_detuning_hz,
                omega_hz,
            ),
            "residual_differential_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.residual_differential_detuning_hz,
                omega_hz,
            ),
            "doppler_detuning_01_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_01_hz,
                omega_hz,
            ),
            "doppler_detuning_11_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_11_hz,
                omega_hz,
            ),
            "extra_rydberg_leakage_rate_dimensionless": self.derived_leakage_rate_dimensionless(),
        }


def yb171_experimental_calibration() -> Yb171ExperimentalCalibration:
    return Yb171ExperimentalCalibration()


def yb171_v4_default_omega_max_hz() -> float:
    return yb171_experimental_calibration().effective_rabi_hz_max


def yb171_gate_time_ns_to_dimensionless(
    gate_time_ns: float,
    *,
    effective_rabi_hz: float | None = None,
) -> float:
    calibration = yb171_experimental_calibration()
    return calibration.physical_gate_time_to_dimensionless(
        float(gate_time_ns) * 1e-9,
        effective_rabi_hz=effective_rabi_hz,
    )


def yb171_dimensionless_time_to_gate_time_ns(
    t_omega: float,
    *,
    effective_rabi_hz: float | None = None,
) -> float:
    calibration = yb171_experimental_calibration()
    return 1e9 * calibration.dimensionless_gate_time_to_seconds(
        float(t_omega),
        effective_rabi_hz=effective_rabi_hz,
    )


def summarize_yb171_v4_result(
    *,
    result,
    gate_time_ns: float,
    omega_max_hz: float,
    model,
) -> dict[str, object]:
    calibration = yb171_experimental_calibration()
    amplitudes = np.asarray(result.amplitudes, dtype=np.float64)
    omega_fraction = np.clip(amplitudes / model.control_amplitude_bound(), 0.0, 1.0)
    effective_omega_hz = omega_fraction * omega_max_hz
    slot_duration_ns = float(gate_time_ns) / int(result.num_tslots)
    slot_midpoints_ns = (np.arange(result.num_tslots, dtype=np.float64) + 0.5) * slot_duration_ns
    return {
        **result.to_json(),
        "gate_time_ns": float(gate_time_ns),
        "gate_time_us": float(gate_time_ns / 1000.0),
        "slot_duration_ns": float(slot_duration_ns),
        "slot_midpoints_ns": [float(x) for x in slot_midpoints_ns],
        "omega_max_hz": float(omega_max_hz),
        "omega_max_mhz": float(omega_max_hz / 1e6),
        "effective_rabi_hz_reference": float(omega_max_hz),
        "effective_rabi_sequence_hz": [float(x) for x in effective_omega_hz],
        "effective_rabi_sequence_mhz": [float(x / 1e6) for x in effective_omega_hz],
        "effective_rabi_fraction": [float(x) for x in omega_fraction],
        "dimensionless_gate_time": float(result.evo_time),
        "physical_time_from_dimensionless_ns": float(
            1e9 * calibration.dimensionless_gate_time_to_seconds(result.evo_time, effective_rabi_hz=omega_max_hz)
        ),
    }


def build_yb171_v3_calibrated_model(
    effective_rabi_hz: float | None = None,
) -> TwoPhotonCZ9DModel:
    calibration = yb171_experimental_calibration()
    return TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        **calibration.nominal_two_photon_parameters(effective_rabi_hz=effective_rabi_hz),
    )


def build_yb171_v4_calibrated_model(
    include_noise: bool = True,
    effective_rabi_hz: float | None = None,
) -> TwoPhotonCZOpen10DModel:
    calibration = yb171_experimental_calibration()
    noise = (
        calibration.open_system_noise(effective_rabi_hz=effective_rabi_hz)
        if include_noise
        else TwoPhotonOpenNoiseConfig()
    )
    return TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        noise=noise,
        **calibration.nominal_two_photon_parameters(effective_rabi_hz=effective_rabi_hz),
    )
