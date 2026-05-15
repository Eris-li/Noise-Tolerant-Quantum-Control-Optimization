from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip

from neutral_yb.config.species import NeutralYb171Species
from neutral_yb.models.evered2023_benchmarking import (
    Evered2023ExponentialBenchmarkResult,
    diagonal_cz_average_gate_fidelity,
    diagonal_cz_process_fidelity,
    evered2023_exponential_decay_fidelity_from_diagonal_map,
)
from neutral_yb.models.global_cz_4d import GlobalCZ4DModel
from neutral_yb.models.two_photon_cz_9d import TwoPhotonCZ9DModel


@dataclass(frozen=True)
class Evered2023TimeOptimalPulse:
    """Fixed-amplitude phase profile from Evered et al., Nature 622, 268 (2023)."""

    amplitude_phase_modulation: float = 2.0 * np.pi * 0.1122
    phase_rate: float = 1.0431
    phase_offset: float = -0.7318
    static_detuning: float = 0.0
    omega_t_over_2pi: float = 1.215

    @property
    def dimensionless_duration(self) -> float:
        return 2.0 * np.pi * self.omega_t_over_2pi

    def phase(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        return (
            self.amplitude_phase_modulation * np.cos(self.phase_rate * times - self.phase_offset)
            + self.static_detuning * times
        )

    def phase_derivative(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        return (
            -self.amplitude_phase_modulation
            * self.phase_rate
            * np.sin(self.phase_rate * times - self.phase_offset)
            + self.static_detuning
        )

    def two_photon_detuning(self, times: np.ndarray) -> np.ndarray:
        """Return the Eq. (2) detuning sign convention, delta(t) = -d phi / dt."""

        return -self.phase_derivative(times)

    def sampled_phases(self, num_tslots: int) -> np.ndarray:
        dt = self.dimensionless_duration / int(num_tslots)
        centers = (np.arange(int(num_tslots), dtype=np.float64) + 0.5) * dt
        return np.mod(self.phase(centers), 2.0 * np.pi)

    def physical_duration_seconds(self, omega_over_2pi_hz: float = 4.6e6) -> float:
        return self.omega_t_over_2pi / float(omega_over_2pi_hz)

    def to_json(self) -> dict[str, float | str]:
        return {
            "source": "Evered et al. Nature 622, 268-272 (2023), Methods Eq. (1)",
            "amplitude_phase_modulation_rad": float(self.amplitude_phase_modulation),
            "amplitude_phase_modulation_over_2pi": float(self.amplitude_phase_modulation / (2.0 * np.pi)),
            "phase_rate_over_omega": float(self.phase_rate),
            "phase_offset_rad": float(self.phase_offset),
            "static_detuning_over_omega": float(self.static_detuning),
            "omega_t_over_2pi": float(self.omega_t_over_2pi),
            "dimensionless_duration_omega_t": float(self.dimensionless_duration),
            "duration_seconds_at_omega_over_2pi_4p6mhz": float(self.physical_duration_seconds()),
        }


@dataclass(frozen=True)
class Evered2023DarkStateConfig:
    """Single-atom three-level ladder parameters for the Methods Eq. (2) model."""

    omega_blue: float
    omega_red: float
    intermediate_detuning: float
    two_photon_detuning: float = 0.0

    def hamiltonian(self) -> qutip.Qobj:
        return qutip.Qobj(
            np.array(
                [
                    [0.0, 0.5 * self.omega_blue, 0.0],
                    [0.5 * self.omega_blue, -self.intermediate_detuning, 0.5 * self.omega_red],
                    [0.0, 0.5 * self.omega_red, -self.two_photon_detuning],
                ],
                dtype=np.complex128,
            )
        )

    def dark_bright_eigenvectors_leading_order(self) -> dict[str, np.ndarray]:
        alpha = self.omega_blue / self.omega_red
        norm = np.sqrt(1.0 + alpha**2)
        delta = self.intermediate_detuning
        omega_r = self.omega_red
        return {
            "D": np.array([-1.0 / norm, 0.0, alpha / norm], dtype=np.complex128),
            "B": np.array(
                [alpha / norm, norm * omega_r / (2.0 * delta), 1.0 / norm],
                dtype=np.complex128,
            ),
            "E": np.array(
                [-alpha * omega_r / (2.0 * delta), 1.0, -omega_r / (2.0 * delta)],
                dtype=np.complex128,
            ),
        }


@dataclass(frozen=True)
class Evered2023ParallelCZCalibration:
    """Default experimental scale factors stated in Evered et al. for CZ gates."""

    atom_species: str = "87Rb"
    rydberg_state_n: int = 53
    omega_over_2pi_hz: float = 4.6e6
    blue_rabi_hz: float = 237.0e6
    red_rabi_hz: float = 303.0e6
    intermediate_detuning_hz: float = 7.8e9
    blockade_shift_hz: float = 450.0e6
    gate_fidelity_parallel_cz: float = 0.995
    max_parallel_atoms: int = 60

    def physical_gate_time_seconds(self, pulse: Evered2023TimeOptimalPulse | None = None) -> float:
        profile = Evered2023TimeOptimalPulse() if pulse is None else pulse
        return profile.physical_duration_seconds(self.omega_over_2pi_hz)

    def to_json(self) -> dict[str, float | int | str]:
        pulse = Evered2023TimeOptimalPulse()
        return {
            "source": "Evered et al. Nature 622, 268-272 (2023)",
            "atom_species": self.atom_species,
            "rydberg_state_n": int(self.rydberg_state_n),
            "omega_over_2pi_hz": float(self.omega_over_2pi_hz),
            "blue_rabi_hz": float(self.blue_rabi_hz),
            "blue_rabi_over_omega": float(self.blue_rabi_hz / self.omega_over_2pi_hz),
            "red_rabi_hz": float(self.red_rabi_hz),
            "red_rabi_over_omega": float(self.red_rabi_hz / self.omega_over_2pi_hz),
            "intermediate_detuning_hz": float(self.intermediate_detuning_hz),
            "intermediate_detuning_over_omega": float(self.intermediate_detuning_hz / self.omega_over_2pi_hz),
            "blockade_shift_hz": float(self.blockade_shift_hz),
            "blockade_shift_over_omega": float(self.blockade_shift_hz / self.omega_over_2pi_hz),
            "time_optimal_gate_time_s": float(self.physical_gate_time_seconds(pulse)),
            "reported_parallel_cz_fidelity": float(self.gate_fidelity_parallel_cz),
            "max_parallel_atoms": int(self.max_parallel_atoms),
            "notes": (
                "The fixed-amplitude exact-gate model is dimensionless. Intermediate detuning, "
                "420/1013 nm Rabi split, finite rise time, measured laser noise, Doppler traces, "
                "and SPAM/readout corrections remain explicit experiment-level inputs."
            ),
        }


@dataclass(frozen=True)
class Evered2023TwoPhotonCZ9DDetuningModel:
    """Two-atom two-photon CZ Hamiltonian in the Evered detuning gauge.

    Basis ordering:
    0. |01>
    1. |0e>
    2. |0r>
    3. |11>
    4. |W_e> = (|1e> + |e1>) / sqrt(2)
    5. |ee>
    6. |W_r> = (|1r> + |r1>) / sqrt(2)
    7. |E_er> = (|er> + |re>) / sqrt(2)
    8. |rr>

    The 1,013-nm leg is fixed, the 420-nm leg has fixed amplitude for the
    time-optimal gate, and the 420-nm phase is represented by the time-dependent
    two-photon detuning ``delta(t) = -d phi / dt``.
    """

    species: NeutralYb171Species
    blue_rabi: float
    red_rabi: float
    intermediate_detuning: float
    blockade_shift: float
    static_resonance_shift: float = 0.0
    use_leading_order_dressed_basis: bool = False

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0e", "0r", "11", "W_e", "ee", "W_r", "E_er", "rr")

    def dimension(self) -> int:
        return 9

    def phase_gate_state_indices(self) -> tuple[int, int]:
        return 0, 3

    def initial_state(self) -> qutip.Qobj:
        branch_01, branch_11 = self.phase_gate_branch_projectors()
        return qutip.Qobj(branch_01 + branch_11)

    def leading_order_blue_dressing_epsilon(self) -> float:
        return float(self.blue_rabi / (2.0 * self.intermediate_detuning))

    def phase_gate_branch_projectors(self) -> tuple[np.ndarray, np.ndarray]:
        branch_01 = np.zeros(9, dtype=np.complex128)
        branch_11 = np.zeros(9, dtype=np.complex128)
        branch_01[0] = 1.0
        branch_11[3] = 1.0
        if self.use_leading_order_dressed_basis:
            epsilon = self.leading_order_blue_dressing_epsilon()
            branch_01[1] = epsilon
            branch_11[4] = np.sqrt(2.0) * epsilon
            branch_11[5] = epsilon**2
            branch_01 /= np.linalg.norm(branch_01)
            branch_11 /= np.linalg.norm(branch_11)
        return branch_01, branch_11

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((9, 9), dtype=np.complex128)
        delta_e = self.intermediate_detuning

        h_d[1, 1] = -delta_e
        h_d[4, 4] = -delta_e
        h_d[5, 5] = -2.0 * delta_e
        h_d[7, 7] = -delta_e
        h_d[8, 8] = self.blockade_shift

        blue_x, _blue_y = self._blue_leg_control_matrices()
        red_x, _red_y = self._red_leg_control_matrices()
        h_d += self.blue_rabi * blue_x + self.red_rabi * red_x
        return qutip.Qobj(h_d)

    def detuning_control_hamiltonian(self) -> qutip.Qobj:
        h_delta = np.zeros((9, 9), dtype=np.complex128)
        h_delta[2, 2] = -1.0
        h_delta[6, 6] = -1.0
        h_delta[7, 7] = -1.0
        h_delta[8, 8] = -2.0
        return qutip.Qobj(h_delta)

    def hamiltonian(self, two_photon_detuning: float) -> qutip.Qobj:
        return self.drift_hamiltonian() + float(two_photon_detuning) * self.detuning_control_hamiltonian()

    def phase_gate_fidelity(self, state: np.ndarray, theta: float) -> float:
        branch_01, branch_11 = self.phase_gate_branch_projectors()
        alpha = complex(np.vdot(branch_01, state))
        beta = complex(np.vdot(branch_11, state))
        return diagonal_cz_process_fidelity(alpha, beta, theta)

    def phase_gate_average_fidelity(self, state: np.ndarray, theta: float) -> float:
        branch_01, branch_11 = self.phase_gate_branch_projectors()
        alpha = complex(np.vdot(branch_01, state))
        beta = complex(np.vdot(branch_11, state))
        return diagonal_cz_average_gate_fidelity(alpha, beta, theta)

    def optimize_theta_for_state(self, state: np.ndarray) -> tuple[float, float]:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda theta: -self.phase_gate_fidelity(state, theta),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.phase_gate_fidelity(state, theta)

    def evered2023_exponential_decay_fidelity(
        self,
        state: np.ndarray,
        theta: float,
        gate_counts: tuple[int, ...] | None = None,
    ) -> Evered2023ExponentialBenchmarkResult:
        branch_01, branch_11 = self.phase_gate_branch_projectors()
        alpha = complex(np.vdot(branch_01, state))
        beta = complex(np.vdot(branch_11, state))
        counts = tuple(range(0, 21, 2)) if gate_counts is None else gate_counts
        return evered2023_exponential_decay_fidelity_from_diagonal_map(alpha, beta, theta, counts)

    @staticmethod
    def _add_quadrature_coupling(
        h_x: np.ndarray,
        h_y: np.ndarray,
        left: int,
        right: int,
        strength: float,
    ) -> None:
        h_x[left, right] = strength
        h_x[right, left] = strength
        h_y[left, right] = -1j * strength
        h_y[right, left] = 1j * strength

    def _blue_leg_control_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        blue_x = np.zeros((9, 9), dtype=np.complex128)
        blue_y = np.zeros((9, 9), dtype=np.complex128)
        self._add_quadrature_coupling(blue_x, blue_y, 0, 1, 0.5)
        self._add_quadrature_coupling(blue_x, blue_y, 3, 4, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(blue_x, blue_y, 4, 5, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(blue_x, blue_y, 6, 7, 0.5)
        return blue_x, blue_y

    def _red_leg_control_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        red_x = np.zeros((9, 9), dtype=np.complex128)
        red_y = np.zeros((9, 9), dtype=np.complex128)
        self._add_quadrature_coupling(red_x, red_y, 1, 2, 0.5)
        self._add_quadrature_coupling(red_x, red_y, 4, 6, 0.5)
        self._add_quadrature_coupling(red_x, red_y, 5, 7, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(red_x, red_y, 7, 8, 1.0 / np.sqrt(2.0))
        return red_x, red_y


def build_evered2023_ideal_global_cz_model(
    species: NeutralYb171Species,
) -> GlobalCZ4DModel:
    """Build the ideal blockade CZ model used by the analytic time-optimal pulse."""

    return GlobalCZ4DModel(species=species)


def build_evered2023_two_photon_ladder_model(
    *,
    species: NeutralYb171Species,
    lower_rabi: float,
    upper_rabi: float,
    intermediate_detuning: float,
    blockade_shift: float,
    two_photon_detuning: float = 0.0,
) -> TwoPhotonCZ9DModel:
    """Build a closed-system two-photon CZ ladder with Evered-style controls."""

    return TwoPhotonCZ9DModel(
        species=species,
        lower_rabi=lower_rabi,
        upper_rabi=upper_rabi,
        intermediate_detuning=intermediate_detuning,
        blockade_shift=blockade_shift,
        two_photon_detuning_01=two_photon_detuning,
        two_photon_detuning_11=two_photon_detuning,
    )


def build_evered2023_two_photon_detuning_model(
    *,
    species: NeutralYb171Species,
    effective_rabi: float = 1.0,
    intermediate_detuning_over_effective_rabi: float | None = None,
    blockade_shift_over_effective_rabi: float | None = None,
    alpha: float = 1.0,
    blue_rabi_over_effective_rabi: float | None = None,
    red_rabi_over_effective_rabi: float | None = None,
    static_resonance_shift: float = 0.0,
    use_leading_order_dressed_basis: bool = False,
) -> Evered2023TwoPhotonCZ9DDetuningModel:
    """Build the two-photon detuning-gauge model in units of the effective Rabi rate."""

    calibration = Evered2023ParallelCZCalibration()
    detuning_ratio = (
        calibration.intermediate_detuning_hz / calibration.omega_over_2pi_hz
        if intermediate_detuning_over_effective_rabi is None
        else float(intermediate_detuning_over_effective_rabi)
    )
    blockade_ratio = (
        calibration.blockade_shift_hz / calibration.omega_over_2pi_hz
        if blockade_shift_over_effective_rabi is None
        else float(blockade_shift_over_effective_rabi)
    )
    if blue_rabi_over_effective_rabi is not None or red_rabi_over_effective_rabi is not None:
        if blue_rabi_over_effective_rabi is None or red_rabi_over_effective_rabi is None:
            raise ValueError("blue_rabi_over_effective_rabi and red_rabi_over_effective_rabi must be provided together")
        blue_rabi = float(blue_rabi_over_effective_rabi) * effective_rabi
        red_rabi = float(red_rabi_over_effective_rabi) * effective_rabi
    else:
        red_rabi = np.sqrt(2.0 * detuning_ratio * effective_rabi**2 / max(float(alpha), 1e-12))
        blue_rabi = float(alpha) * red_rabi
    return Evered2023TwoPhotonCZ9DDetuningModel(
        species=species,
        blue_rabi=float(blue_rabi),
        red_rabi=float(red_rabi),
        intermediate_detuning=float(detuning_ratio * effective_rabi),
        blockade_shift=float(blockade_ratio * effective_rabi),
        static_resonance_shift=float(static_resonance_shift),
        use_leading_order_dressed_basis=bool(use_leading_order_dressed_basis),
    )
