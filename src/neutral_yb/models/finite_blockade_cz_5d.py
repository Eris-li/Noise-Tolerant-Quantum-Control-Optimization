from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip
from scipy.optimize import minimize_scalar

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class FiniteBlockadeCZ5DModel:
    """Closed-system 5D model for a nonideal global CZ gate.

    Basis ordering:
    0. |01>
    1. |0r>
    2. |11>
    3. |W> = (|1r> + |r1>) / sqrt(2)
    4. |rr>

    This model keeps the symmetry structure of the ideal global CZ problem,
    but introduces the leading Hamiltonian-level imperfections:
    finite blockade, static detuning, and multiplicative Rabi mismatch.
    """

    species: NeutralYb171Species
    blockade_shift: float
    static_detuning_01: float = 0.0
    static_detuning_11: float = 0.0
    rabi_scale: float = 1.0

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0r", "11", "W", "rr")

    def dimension(self) -> int:
        return 5

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((5, 5), dtype=np.complex128)
        h_d[1, 1] = -self.static_detuning_01
        h_d[3, 3] = -self.static_detuning_11
        h_d[4, 4] = self.blockade_shift - 2.0 * self.static_detuning_11
        return qutip.Qobj(h_d)

    def control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x = np.zeros((5, 5), dtype=np.complex128)
        h_y = np.zeros((5, 5), dtype=np.complex128)

        scale = self.rabi_scale
        self._add_quadrature_coupling(h_x, h_y, 0, 1, 0.5 * scale)
        self._add_quadrature_coupling(h_x, h_y, 2, 3, (1.0 / np.sqrt(2.0)) * scale)
        self._add_quadrature_coupling(h_x, h_y, 3, 4, (1.0 / np.sqrt(2.0)) * scale)

        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def initial_state(self) -> qutip.Qobj:
        return qutip.Qobj(np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.complex128))

    def phase_gate_fidelity(self, state: np.ndarray, theta: float) -> float:
        alpha = complex(state[0])
        beta = complex(state[2])
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
                [np.exp(1j * theta), 0.0, -np.exp(2j * theta), 0.0, 0.0],
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
