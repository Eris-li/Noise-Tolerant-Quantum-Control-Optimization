from __future__ import annotations

from dataclasses import asdict, dataclass
import math

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

    def omega_ref_rad_s(self) -> float:
        return 2.0 * math.pi * self.effective_rabi_hz

    def dimensionless_hamiltonian_frequency(self, hz: float) -> float:
        return hz / self.effective_rabi_hz

    def dimensionless_rate_from_lifetime(self, lifetime_s: float) -> float:
        return 1.0 / (self.omega_ref_rad_s() * lifetime_s)

    def derived_one_photon_rabi_hz(self) -> float:
        return math.sqrt(2.0 * self.intermediate_detuning_hz * self.effective_rabi_hz)

    def derived_one_photon_rabi_dimensionless(self) -> float:
        return self.dimensionless_hamiltonian_frequency(self.derived_one_photon_rabi_hz())

    def derived_leakage_rate_dimensionless(self) -> float:
        return self.unwanted_mf_error_per_gate / self.leakage_reference_t_omega

    def nominal_two_photon_parameters(self) -> dict[str, float]:
        lower_upper = self.derived_one_photon_rabi_dimensionless()
        return {
            "lower_rabi": lower_upper,
            "upper_rabi": lower_upper,
            "intermediate_detuning": self.dimensionless_hamiltonian_frequency(self.intermediate_detuning_hz),
            "blockade_shift": self.dimensionless_hamiltonian_frequency(self.blockade_shift_hz),
            "two_photon_detuning_01": 0.0,
            "two_photon_detuning_11": 0.0,
        }

    def open_system_noise(self) -> TwoPhotonOpenNoiseConfig:
        # The explicit intermediate state is only a ladder surrogate. The latest
        # high-fidelity 171Yb gates use a single-photon clock-to-Rydberg drive,
        # so we do not assign a literal spontaneous-scattering rate here.
        return TwoPhotonOpenNoiseConfig(
            intermediate_detuning_offset=0.0,
            common_two_photon_detuning=self.dimensionless_hamiltonian_frequency(self.residual_common_detuning_hz),
            differential_two_photon_detuning=self.dimensionless_hamiltonian_frequency(
                self.residual_differential_detuning_hz
            ),
            doppler_detuning_01=self.dimensionless_hamiltonian_frequency(self.doppler_detuning_01_hz),
            doppler_detuning_11=self.dimensionless_hamiltonian_frequency(self.doppler_detuning_11_hz),
            blockade_shift_offset=0.0,
            lower_amplitude_scale=1.0 - self.uv_pulse_area_fractional_error,
            upper_amplitude_scale=1.0 - self.uv_pulse_area_fractional_error,
            intermediate_decay_rate=0.0,
            rydberg_decay_rate=self.dimensionless_rate_from_lifetime(self.rydberg_lifetime_s),
            intermediate_dephasing_rate=0.0,
            rydberg_dephasing_rate=self.dimensionless_rate_from_lifetime(self.rydberg_t2_star_s),
            extra_rydberg_leakage_rate=self.derived_leakage_rate_dimensionless(),
            intermediate_branch_to_qubit=1.0,
            rydberg_branch_to_qubit=0.0,
        )

    def summary(self) -> dict[str, object]:
        one_photon_rabi_hz = self.derived_one_photon_rabi_hz()
        return {
            **asdict(self),
            "omega_ref_rad_s": self.omega_ref_rad_s(),
            "derived_one_photon_rabi_hz": one_photon_rabi_hz,
            "derived_one_photon_rabi_over_omega_ref": self.derived_one_photon_rabi_dimensionless(),
            "intermediate_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.intermediate_detuning_hz
            ),
            "blockade_shift_dimensionless": self.dimensionless_hamiltonian_frequency(self.blockade_shift_hz),
            "rydberg_decay_rate_dimensionless": self.dimensionless_rate_from_lifetime(self.rydberg_lifetime_s),
            "rydberg_dephasing_rate_dimensionless": self.dimensionless_rate_from_lifetime(self.rydberg_t2_star_s),
            "residual_common_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.residual_common_detuning_hz
            ),
            "residual_differential_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.residual_differential_detuning_hz
            ),
            "doppler_detuning_01_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_01_hz
            ),
            "doppler_detuning_11_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.doppler_detuning_11_hz
            ),
            "extra_rydberg_leakage_rate_dimensionless": self.derived_leakage_rate_dimensionless(),
        }


def yb171_experimental_calibration() -> Yb171ExperimentalCalibration:
    return Yb171ExperimentalCalibration()


def build_yb171_v3_calibrated_model() -> TwoPhotonCZ9DModel:
    calibration = yb171_experimental_calibration()
    return TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        **calibration.nominal_two_photon_parameters(),
    )


def build_yb171_v4_calibrated_model(
    include_noise: bool = True,
) -> TwoPhotonCZOpen10DModel:
    calibration = yb171_experimental_calibration()
    noise = calibration.open_system_noise() if include_noise else TwoPhotonOpenNoiseConfig()
    return TwoPhotonCZOpen10DModel(
        species=idealised_yb171(),
        noise=noise,
        **calibration.nominal_two_photon_parameters(),
    )
