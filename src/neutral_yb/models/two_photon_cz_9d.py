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

    The Hamiltonian uses a phase-modulated lower-leg optical coupling and a
    fixed-phase upper-leg coupling. This matches the common experimental gauge
    choice where only one optical phase is optimized explicitly and the second
    leg is kept as a phase reference.
    """

    species: NeutralYb171Species
    lower_rabi: float
    upper_rabi: float
    intermediate_detuning: float
    blockade_shift: float
    two_photon_detuning_01: float = 0.0
    two_photon_detuning_11: float = 0.0
    upper_leg_phase: float = 0.0

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0e", "0r", "11", "W_e", "ee", "W_r", "E_er", "rr")

    def dimension(self) -> int:
        return 9

    def phase_gate_state_indices(self) -> tuple[int, int]:
        return 0, 3

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

        upper_x, upper_y = self._upper_leg_control_matrices()
        h_d += self.upper_rabi * (
            np.cos(self.upper_leg_phase) * upper_x + np.sin(self.upper_leg_phase) * upper_y
        )
        return qutip.Qobj(h_d)

    def phase_control_hamiltonians(self) -> tuple[tuple[qutip.Qobj, qutip.Qobj], ...]:
        lower_x = np.zeros((9, 9), dtype=np.complex128)
        lower_y = np.zeros((9, 9), dtype=np.complex128)

        self._add_quadrature_coupling(lower_x, lower_y, 0, 1, 0.5)
        self._add_quadrature_coupling(lower_x, lower_y, 3, 4, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(lower_x, lower_y, 4, 5, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(lower_x, lower_y, 6, 7, 0.5)

        return ((qutip.Qobj(lower_x), qutip.Qobj(lower_y)),)

    def phase_control_amplitudes(self) -> tuple[float]:
        return (self.lower_rabi,)

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

    def _upper_leg_control_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        upper_x = np.zeros((9, 9), dtype=np.complex128)
        upper_y = np.zeros((9, 9), dtype=np.complex128)
        self._add_quadrature_coupling(upper_x, upper_y, 1, 2, 0.5)
        self._add_quadrature_coupling(upper_x, upper_y, 4, 6, 0.5)
        self._add_quadrature_coupling(upper_x, upper_y, 5, 7, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(upper_x, upper_y, 7, 8, 1.0 / np.sqrt(2.0))
        return upper_x, upper_y
