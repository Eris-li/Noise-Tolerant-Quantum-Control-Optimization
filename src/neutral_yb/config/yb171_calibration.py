from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Literal

import numpy as np

from neutral_yb.config.species import idealised_yb171
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel
from neutral_yb.models.yb171_clock_rydberg_cz_open import (
    Yb171ClockRydbergCZOpenModel,
    Yb171ClockRydbergNoiseConfig,
)

Yb171CalibrationProfile = Literal["strict_literature_minimal", "experimental_surrogate_full"]


@dataclass(frozen=True)
class Yb171ExperimentalCalibration:
    """^171Yb clock-to-Rydberg calibration anchored to recent APS results.

    Physical picture:
    - qubit states live in the ground-state nuclear-spin manifold
    - a gate begins by shelving logical ``|1>`` into the metastable clock state
      ``|c> = |3P0, m_F=-1/2>``
    - a single 301.9-302 nm UV beam drives ``|c> <-> |r>`` and implements the
      entangling segment through the Rydberg blockade mechanism
    - ideal unshelving maps ``|c>`` back into the logical ``|1>`` state

    Sources:
    - Muniz et al., PRX Quantum 6, 020334 (2025)
    - Peper et al., Phys. Rev. X 15, 011009 (2025)
    """

    uv_rabi_hz: float = 10e6
    uv_rabi_hz_max: float = 10e6
    clock_shelving_rabi_hz: float = 7e3
    clock_pi_pulse_duration_s: float = 130e-6
    clock_num_steps: int = 16
    blockade_shift_hz: float = 160e6
    clock_state_lifetime_s: float = 1.06
    clock_trap_loss_lifetime_s: float = 5.0
    rydberg_lifetime_s: float = 65e-6
    rydberg_t2_star_s: float = 3.4e-6
    rydberg_t2_echo_s: float = 5.1e-6
    clock_pulse_area_fractional_rms: float = 0.0
    clock_lamb_dicke_eta: float = 0.26
    clock_temperature_nbar: float = 0.25
    quasistatic_clock_detuning_rms_hz: float = 33.0
    differential_clock_detuning_rms_hz: float = 0.0
    clock_phase_noise_psd_fmin_hz: float = 10.0
    clock_phase_noise_psd_fmax_hz: float = 50e3
    clock_phase_noise_num_bins: int = 128
    clock_phase_noise_psd_level_rad2_per_hz: float = 2.0e-8
    uv_pulse_area_fractional_rms: float = 0.004
    quasistatic_uv_detuning_rms_hz: float | None = None
    differential_uv_detuning_rms_hz: float = 0.0
    blockade_shift_jitter_hz: float = 0.0
    markovian_clock_dephasing_t2_s: float | None = None
    markovian_rydberg_dephasing_t2_s: float | None = None
    neighboring_mf_leakage_per_gate: float = 0.0
    leakage_reference_t_omega: float = 8.0
    clock_decay_as_single_loss_channel: bool = False
    profile_name: str = "experimental_surrogate_full"
    rydberg_state_label: str = "|65 3S1, F=1/2, m_F=-1/2>"
    clock_state_label: str = "|3P0, m_F=-1/2>"

    @property
    def effective_rabi_hz(self) -> float:
        return self.uv_rabi_hz

    @property
    def effective_rabi_hz_max(self) -> float:
        return self.uv_rabi_hz_max

    def resolve_effective_rabi_hz(self, effective_rabi_hz: float | None = None) -> float:
        return float(self.uv_rabi_hz if effective_rabi_hz is None else effective_rabi_hz)

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

    def resolved_quasistatic_uv_detuning_rms_hz(self) -> float:
        if self.quasistatic_uv_detuning_rms_hz is not None:
            return float(self.quasistatic_uv_detuning_rms_hz)
        total_scale_hz = float(1.0 / (2.0 * math.pi * self.rydberg_t2_star_s))
        if self.rydberg_t2_echo_s <= 0.0:
            return total_scale_hz
        fast_scale_hz = float(1.0 / (2.0 * math.pi * self.rydberg_t2_echo_s))
        return float(math.sqrt(max(total_scale_hz * total_scale_hz - fast_scale_hz * fast_scale_hz, 0.0)))

    def derived_rydberg_pure_dephasing_time_s(self) -> float | None:
        if self.rydberg_t2_echo_s <= 0.0 or self.rydberg_lifetime_s <= 0.0:
            return None
        inverse_tphi = (1.0 / float(self.rydberg_t2_echo_s)) - (0.5 / float(self.rydberg_lifetime_s))
        if inverse_tphi <= 0.0:
            return None
        return float(1.0 / inverse_tphi)

    def clock_total_num_steps(self) -> int:
        return int(self.clock_num_steps)

    def clock_total_duration_s(self) -> float:
        return float(self.clock_pi_pulse_duration_s)

    def _clock_thermal_carrier_reduction(self, motional_n: int) -> float:
        eta_sq = float(self.clock_lamb_dicke_eta) ** 2
        coeffs = np.zeros(int(max(motional_n, 0)) + 1, dtype=np.float64)
        coeffs[int(max(motional_n, 0))] = 1.0
        laguerre_value = np.polynomial.laguerre.lagval(eta_sq, coeffs)
        return float(np.exp(-0.5 * eta_sq) * laguerre_value)

    def _clock_thermal_average_reduction(self) -> float:
        nbar = float(max(self.clock_temperature_nbar, 0.0))
        if nbar <= 0.0:
            return self._clock_thermal_carrier_reduction(0)
        probability = 1.0 / (1.0 + nbar)
        total = 0.0
        for motional_n in range(256):
            total += probability * self._clock_thermal_carrier_reduction(motional_n)
            probability *= nbar / (1.0 + nbar)
            if probability < 1e-12:
                break
        return float(max(total, 1e-12))

    def sample_clock_thermal_amplitude_scale(self, *, rng: np.random.Generator) -> float:
        nbar = float(max(self.clock_temperature_nbar, 0.0))
        if nbar <= 0.0:
            return 1.0
        probability = 1.0 / (1.0 + nbar)
        motional_n = int(rng.geometric(probability) - 1)
        reduction = self._clock_thermal_carrier_reduction(motional_n)
        average = self._clock_thermal_average_reduction()
        return float(max(reduction / average, 0.0))

    def sample_clock_phase_noise_trace(
        self,
        *,
        rng: np.random.Generator,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.clock_phase_noise_psd_level_rad2_per_hz <= 0.0:
            zeros = tuple(0.0 for _ in range(self.clock_total_num_steps()))
            return zeros, zeros

        frequencies = np.linspace(
            float(self.clock_phase_noise_psd_fmin_hz),
            float(self.clock_phase_noise_psd_fmax_hz),
            int(max(self.clock_phase_noise_num_bins, 2)),
            dtype=np.float64,
        )
        delta_f = float((frequencies[-1] - frequencies[0]) / (len(frequencies) - 1))
        amplitude = 2.0 * np.sqrt(float(self.clock_phase_noise_psd_level_rad2_per_hz) * delta_f)
        num_steps = self.clock_total_num_steps()
        duration_s = self.clock_total_duration_s()
        sample_times = (np.arange(num_steps, dtype=np.float64) + 0.5) * (duration_s / float(num_steps))

        def sample_trace() -> tuple[float, ...]:
            phases = rng.uniform(0.0, 2.0 * math.pi, size=frequencies.size)
            values = np.zeros_like(sample_times)
            for frequency_hz, random_phase in zip(frequencies, phases):
                values += amplitude * np.cos(2.0 * math.pi * frequency_hz * sample_times + random_phase)
            return tuple(float(value) for value in values)

        return sample_trace(), sample_trace()

    def derived_neighboring_mf_leakage_rate_dimensionless(self) -> float:
        return self.neighboring_mf_leakage_per_gate / self.leakage_reference_t_omega

    def nominal_clock_rydberg_parameters(
        self,
        effective_rabi_hz: float | None = None,
    ) -> dict[str, float]:
        return {
            "uv_rabi": self.dimensionless_hamiltonian_frequency(
                self.resolve_effective_rabi_hz(effective_rabi_hz),
                effective_rabi_hz,
            ),
            "blockade_shift": self.dimensionless_hamiltonian_frequency(
                self.blockade_shift_hz,
                effective_rabi_hz,
            ),
            "clock_pi_time": self.physical_gate_time_to_dimensionless(
                self.clock_pi_pulse_duration_s,
                effective_rabi_hz=effective_rabi_hz,
            ),
            "clock_num_steps": int(self.clock_num_steps),
            "static_clock_detuning_01": 0.0,
            "static_clock_detuning_11": 0.0,
            "static_uv_detuning_01": 0.0,
            "static_uv_detuning_11": 0.0,
        }

    def open_system_noise(
        self,
        effective_rabi_hz: float | None = None,
    ) -> Yb171ClockRydbergNoiseConfig:
        clock_dephasing_rate = 0.0
        if self.markovian_clock_dephasing_t2_s is not None:
            clock_dephasing_rate = self.dimensionless_rate_from_lifetime(
                self.markovian_clock_dephasing_t2_s,
                effective_rabi_hz,
            )
        rydberg_dephasing_rate = 0.0
        if self.markovian_rydberg_dephasing_t2_s is not None:
            rydberg_dephasing_rate = self.dimensionless_rate_from_lifetime(
                self.markovian_rydberg_dephasing_t2_s,
                effective_rabi_hz,
            )
        return Yb171ClockRydbergNoiseConfig(
            common_clock_detuning=0.0,
            differential_clock_detuning=0.0,
            common_uv_detuning=0.0,
            differential_uv_detuning=0.0,
            blockade_shift_offset=0.0,
            clock_amplitude_scale=1.0,
            uv_amplitude_scale=1.0,
            clock_decay_rate=(
                self.dimensionless_rate_from_lifetime(
                    self.clock_state_lifetime_s,
                    effective_rabi_hz,
                )
                if self.clock_decay_as_single_loss_channel
                else 0.0
            ),
            clock_scattering_rate=(
                0.0
                if self.clock_decay_as_single_loss_channel
                else self.dimensionless_rate_from_lifetime(
                    self.clock_state_lifetime_s,
                    effective_rabi_hz,
                )
            ),
            clock_loss_rate=(
                0.0
                if self.clock_decay_as_single_loss_channel
                else self.dimensionless_rate_from_lifetime(
                    self.clock_trap_loss_lifetime_s,
                    effective_rabi_hz,
                )
            ),
            clock_dephasing_rate=clock_dephasing_rate,
            rydberg_decay_rate=self.dimensionless_rate_from_lifetime(
                self.rydberg_lifetime_s,
                effective_rabi_hz,
            ),
            rydberg_dephasing_rate=rydberg_dephasing_rate,
            neighboring_mf_leakage_rate=self.derived_neighboring_mf_leakage_rate_dimensionless(),
        )

    def sample_quasistatic_noise(
        self,
        *,
        rng: np.random.Generator,
        effective_rabi_hz: float | None = None,
    ) -> Yb171ClockRydbergNoiseConfig:
        nominal = self.open_system_noise(effective_rabi_hz=effective_rabi_hz)
        sampled_clock_scale = self.sample_clock_thermal_amplitude_scale(rng=rng)
        if self.clock_pulse_area_fractional_rms > 0.0:
            sampled_clock_scale *= float(rng.normal(1.0, self.clock_pulse_area_fractional_rms))
        sampled_scale = float(rng.normal(1.0, self.uv_pulse_area_fractional_rms))
        prefix_phase_trace, suffix_phase_trace = self.sample_clock_phase_noise_trace(rng=rng)
        return replace(
            nominal,
            common_clock_detuning=self.dimensionless_hamiltonian_frequency(
                float(rng.normal(0.0, self.quasistatic_clock_detuning_rms_hz)),
                effective_rabi_hz,
            ),
            differential_clock_detuning=self.dimensionless_hamiltonian_frequency(
                float(rng.normal(0.0, self.differential_clock_detuning_rms_hz)),
                effective_rabi_hz,
            ),
            common_uv_detuning=self.dimensionless_hamiltonian_frequency(
                float(rng.normal(0.0, self.resolved_quasistatic_uv_detuning_rms_hz())),
                effective_rabi_hz,
            ),
            differential_uv_detuning=self.dimensionless_hamiltonian_frequency(
                float(rng.normal(0.0, self.differential_uv_detuning_rms_hz)),
                effective_rabi_hz,
            ),
            blockade_shift_offset=self.dimensionless_hamiltonian_frequency(
                float(rng.normal(0.0, self.blockade_shift_jitter_hz)),
                effective_rabi_hz,
            ),
            clock_amplitude_scale=max(0.0, sampled_clock_scale),
            uv_amplitude_scale=max(0.0, sampled_scale),
            clock_phase_trace_prefix=prefix_phase_trace,
            clock_phase_trace_suffix=suffix_phase_trace,
        )

    def summary(self, effective_rabi_hz: float | None = None) -> dict[str, object]:
        omega_hz = self.resolve_effective_rabi_hz(effective_rabi_hz)
        return {
            **asdict(self),
            "calibration_reference_primary": "Muniz et al., PRX Quantum 6, 020334 (2025)",
            "calibration_reference_interaction": "Peper et al., Phys. Rev. X 15, 011009 (2025)",
            "model_kind": "effective full ^171Yb clock-to-Rydberg CZ gate",
            "profile_name": self.profile_name,
            "resolved_uv_rabi_hz": omega_hz,
            "resolved_uv_rabi_mhz": omega_hz / 1e6,
            "clock_pi_pulse_duration_us": self.clock_pi_pulse_duration_s * 1e6,
            "clock_num_steps": int(self.clock_num_steps),
            "clock_pi_time_dimensionless": self.physical_gate_time_to_dimensionless(
                self.clock_pi_pulse_duration_s,
                effective_rabi_hz=omega_hz,
            ),
            "clock_decay_rate_dimensionless": 0.0,
            "clock_scattering_rate_dimensionless": self.dimensionless_rate_from_lifetime(
                self.clock_state_lifetime_s,
                omega_hz,
            ),
            "clock_loss_rate_dimensionless": self.dimensionless_rate_from_lifetime(
                self.clock_trap_loss_lifetime_s,
                omega_hz,
            ),
            "clock_dephasing_rate_dimensionless": (
                0.0
                if self.markovian_clock_dephasing_t2_s is None
                else self.dimensionless_rate_from_lifetime(
                    self.markovian_clock_dephasing_t2_s,
                    omega_hz,
                )
            ),
            "uv_rabi_dimensionless": self.dimensionless_hamiltonian_frequency(omega_hz, omega_hz),
            "blockade_shift_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.blockade_shift_hz,
                omega_hz,
            ),
            "rydberg_decay_rate_dimensionless": self.dimensionless_rate_from_lifetime(
                self.rydberg_lifetime_s,
                omega_hz,
            ),
            "rydberg_dephasing_rate_dimensionless": (
                0.0
                if self.markovian_rydberg_dephasing_t2_s is None
                else self.dimensionless_rate_from_lifetime(
                    self.markovian_rydberg_dephasing_t2_s,
                    omega_hz,
                )
            ),
            "resolved_quasistatic_uv_detuning_rms_hz": self.resolved_quasistatic_uv_detuning_rms_hz(),
            "derived_rydberg_pure_dephasing_time_s": self.derived_rydberg_pure_dephasing_time_s(),
            "resolved_quasistatic_clock_detuning_rms_hz": float(self.quasistatic_clock_detuning_rms_hz),
            "resolved_quasistatic_uv_detuning_dimensionless": self.dimensionless_hamiltonian_frequency(
                self.resolved_quasistatic_uv_detuning_rms_hz(),
                omega_hz,
            ),
            "derived_rydberg_pure_dephasing_rate_dimensionless": (
                0.0
                if self.derived_rydberg_pure_dephasing_time_s() is None
                else self.dimensionless_rate_from_lifetime(
                    self.derived_rydberg_pure_dephasing_time_s(),
                    omega_hz,
                )
            ),
            "clock_total_duration_us": self.clock_total_duration_s() * 1e6,
            "clock_total_num_steps": self.clock_total_num_steps(),
            "neighboring_mf_leakage_rate_dimensionless": self.derived_neighboring_mf_leakage_rate_dimensionless(),
}


def yb171_experimental_calibration(
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> Yb171ExperimentalCalibration:
    base = Yb171ExperimentalCalibration()
    if profile == "experimental_surrogate_full":
        return replace(
            base,
            markovian_rydberg_dephasing_t2_s=base.derived_rydberg_pure_dephasing_time_s(),
            profile_name=profile,
        )
    if profile == "strict_literature_minimal":
        return replace(
            base,
            clock_trap_loss_lifetime_s=math.inf,
            quasistatic_uv_detuning_rms_hz=0.0,
            clock_phase_noise_psd_level_rad2_per_hz=0.0,
            clock_decay_as_single_loss_channel=True,
            profile_name=profile,
        )
    raise ValueError(f"Unsupported Yb171 calibration profile: {profile}")


def yb171_v4_default_omega_max_hz() -> float:
    return yb171_experimental_calibration().uv_rabi_hz_max


def yb171_v5_default_omega_max_hz(
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> float:
    return yb171_experimental_calibration(profile=profile).uv_rabi_hz_max


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
    prefix_time = float(model.fixed_prefix_duration()) if hasattr(model, "fixed_prefix_duration") else calibration.physical_gate_time_to_dimensionless(
        calibration.clock_pi_pulse_duration_s,
        effective_rabi_hz=omega_max_hz,
    )
    suffix_time = float(model.fixed_suffix_duration()) if hasattr(model, "fixed_suffix_duration") else prefix_time
    prefix_ns = 1e9 * calibration.dimensionless_gate_time_to_seconds(prefix_time, effective_rabi_hz=omega_max_hz)
    suffix_ns = 1e9 * calibration.dimensionless_gate_time_to_seconds(suffix_time, effective_rabi_hz=omega_max_hz)
    return {
        **result.to_json(),
        "phase_gate_fidelity": float(result.probe_fidelity),
        "fidelity_metric": "paper_eq7_special_state_phase_gate_fidelity",
        "uv_segment_time_ns": float(gate_time_ns),
        "uv_segment_time_us": float(gate_time_ns / 1000.0),
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
        "clock_pi_pulse_duration_us": float(calibration.clock_pi_pulse_duration_s * 1e6),
        "clock_pi_pulse_duration_ns": float(calibration.clock_pi_pulse_duration_s * 1e9),
        "fixed_prefix_duration_us": float(prefix_ns / 1000.0),
        "fixed_prefix_duration_ns": float(prefix_ns),
        "fixed_suffix_duration_us": float(suffix_ns / 1000.0),
        "fixed_suffix_duration_ns": float(suffix_ns),
        "clock_num_steps": int(getattr(model, "clock_num_steps", calibration.clock_num_steps)),
        "total_gate_time_ns": float(
            gate_time_ns + prefix_ns + suffix_ns
        ),
        "total_gate_time_us": float(
            gate_time_ns / 1000.0 + prefix_ns / 1000.0 + suffix_ns / 1000.0
        ),
        "basis_labels": list(model.basis_labels()),
        "model_kind": "effective full ^171Yb clock-to-Rydberg CZ gate",
    }


def build_yb171_v3_calibrated_model(
    effective_rabi_hz: float | None = None,
) -> TwoPhotonCZ9DModel:
    calibration = yb171_experimental_calibration()
    # v3 remains the historical ladder reference and is intentionally unchanged.
    # A small compatibility mapping is kept here for side-by-side legacy studies.
    effective_hz = calibration.resolve_effective_rabi_hz(effective_rabi_hz)
    surrogate_rabi = math.sqrt(2.0 * 7.8e9 * effective_hz)
    return TwoPhotonCZ9DModel(
        species=idealised_yb171(),
        lower_rabi=surrogate_rabi / effective_hz,
        upper_rabi=surrogate_rabi / effective_hz,
        intermediate_detuning=7.8e9 / effective_hz,
        blockade_shift=calibration.blockade_shift_hz / effective_hz,
    )


def build_yb171_v4_calibrated_model(
    include_noise: bool = True,
    effective_rabi_hz: float | None = None,
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> Yb171ClockRydbergCZOpenModel:
    calibration = yb171_experimental_calibration(profile=profile)
    noise = (
        calibration.open_system_noise(effective_rabi_hz=effective_rabi_hz)
        if include_noise
        else Yb171ClockRydbergNoiseConfig()
    )
    return Yb171ClockRydbergCZOpenModel(
        species=idealised_yb171(),
        noise=noise,
        **calibration.nominal_clock_rydberg_parameters(effective_rabi_hz=effective_rabi_hz),
    )


def build_yb171_v4_quasistatic_ensemble(
    *,
    ensemble_size: int,
    seed: int = 17,
    include_noise: bool = True,
    effective_rabi_hz: float | None = None,
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> list[Yb171ClockRydbergCZOpenModel]:
    calibration = yb171_experimental_calibration(profile=profile)
    rng = np.random.default_rng(seed)
    models: list[Yb171ClockRydbergCZOpenModel] = []
    for _ in range(int(max(ensemble_size, 1))):
        noise = (
            calibration.sample_quasistatic_noise(
                rng=rng,
                effective_rabi_hz=effective_rabi_hz,
            )
            if include_noise
            else Yb171ClockRydbergNoiseConfig()
        )
        models.append(
            Yb171ClockRydbergCZOpenModel(
                species=idealised_yb171(),
                noise=noise,
                **calibration.nominal_clock_rydberg_parameters(effective_rabi_hz=effective_rabi_hz),
            )
        )
    return models


def build_yb171_v5_calibrated_model(
    include_noise: bool = True,
    effective_rabi_hz: float | None = None,
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> Yb171ClockRydbergCZOpenModel:
    return build_yb171_v4_calibrated_model(
        include_noise=include_noise,
        effective_rabi_hz=effective_rabi_hz,
        profile=profile,
    )


def build_yb171_v5_quasistatic_ensemble(
    *,
    ensemble_size: int,
    seed: int = 17,
    include_noise: bool = True,
    effective_rabi_hz: float | None = None,
    profile: Yb171CalibrationProfile = "experimental_surrogate_full",
) -> list[Yb171ClockRydbergCZOpenModel]:
    return build_yb171_v4_quasistatic_ensemble(
        ensemble_size=ensemble_size,
        seed=seed,
        include_noise=include_noise,
        effective_rabi_hz=effective_rabi_hz,
        profile=profile,
    )


def summarize_yb171_v5_result(
    *,
    result,
    gate_time_ns: float,
    omega_max_hz: float,
    model,
) -> dict[str, object]:
    payload = summarize_yb171_v4_result(
        result=result,
        gate_time_ns=gate_time_ns,
        omega_max_hz=omega_max_hz,
        model=model,
    )
    payload["model_version"] = "v5"
    return payload
