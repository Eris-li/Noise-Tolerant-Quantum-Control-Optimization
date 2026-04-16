from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip
from scipy.optimize import minimize_scalar

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class TwoPhotonCZ9DModel:
    """Symmetry-reduced closed-system CZ model with an explicit intermediate state.

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

    The Hamiltonian is written in a frame where the two optical couplings are
    real and fixed. The time-dependent controls are the two laser phase rates,
    which enter as diagonal frequency shifts. The integrated controls can be
    interpreted as the two optical phase sequences to compare across pulses.
    """

    species: NeutralYb171Species
    lower_rabi: float
    upper_rabi: float
    intermediate_detuning: float
    blockade_shift: float
    two_photon_detuning_01: float = 0.0
    two_photon_detuning_11: float = 0.0

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0e", "0r", "11", "W_e", "ee", "W_r", "E_er", "rr")

    def dimension(self) -> int:
        return 9

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((9, 9), dtype=np.complex128)
        delta = self.intermediate_detuning
        det_01 = self.two_photon_detuning_01
        det_11 = self.two_photon_detuning_11

        h_d[1, 1] = -delta
        h_d[2, 2] = -det_01
        h_d[4, 4] = -delta
        h_d[5, 5] = -2.0 * delta
        h_d[6, 6] = -det_11
        h_d[7, 7] = -(delta + det_11)
        h_d[8, 8] = self.blockade_shift - 2.0 * det_11

        self._add_real_coupling(h_d, 0, 1, 0.5 * self.lower_rabi)
        self._add_real_coupling(h_d, 3, 4, (1.0 / np.sqrt(2.0)) * self.lower_rabi)
        self._add_real_coupling(h_d, 4, 5, (1.0 / np.sqrt(2.0)) * self.lower_rabi)
        self._add_real_coupling(h_d, 6, 7, 0.5 * self.lower_rabi)

        self._add_real_coupling(h_d, 1, 2, 0.5 * self.upper_rabi)
        self._add_real_coupling(h_d, 4, 6, 0.5 * self.upper_rabi)
        self._add_real_coupling(h_d, 5, 7, (1.0 / np.sqrt(2.0)) * self.upper_rabi)
        self._add_real_coupling(h_d, 7, 8, (1.0 / np.sqrt(2.0)) * self.upper_rabi)
        return qutip.Qobj(h_d)

    def control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        lower_phase_rate = np.diag([0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0])
        upper_phase_rate = np.diag([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        return qutip.Qobj(lower_phase_rate), qutip.Qobj(upper_phase_rate)

    def initial_state(self) -> qutip.Qobj:
        return qutip.Qobj(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128))

    def phase_gate_fidelity(self, state: np.ndarray, theta: float) -> float:
        alpha = complex(state[0])
        beta = complex(state[3])
        phased_sum = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
        population_sum = 1.0 + 2.0 * abs(alpha) ** 2 + abs(beta) ** 2
        return float((abs(phased_sum) ** 2 + population_sum) / 20.0)

    def optimize_theta_for_state(self, state: np.ndarray) -> tuple[float, float]:
        result = minimize_scalar(
            lambda theta: -self.phase_gate_fidelity(state, theta),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.phase_gate_fidelity(state, theta)

    def target_state(self, theta: float) -> qutip.Qobj:
        return qutip.Qobj(
            np.array(
                [np.exp(1j * theta), 0.0, 0.0, -np.exp(2j * theta), 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.complex128,
            )
        )

    @staticmethod
    def _add_real_coupling(matrix: np.ndarray, left: int, right: int, strength: float) -> None:
        matrix[left, right] = strength
        matrix[right, left] = strength
