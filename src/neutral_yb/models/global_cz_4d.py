from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip
from scipy.optimize import minimize_scalar

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class GlobalCZ4DModel:
    """4D symmetry-reduced ideal global CZ model following arXiv:2202.00903."""

    species: NeutralYb171Species

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0r", "11", "W")

    def dimension(self) -> int:
        return 4

    def drift_hamiltonian(self) -> qutip.Qobj:
        return qutip.Qobj(np.zeros((4, 4), dtype=np.complex128))

    def control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x = np.zeros((4, 4), dtype=np.complex128)
        h_y = np.zeros((4, 4), dtype=np.complex128)

        self._add_quadrature_coupling(h_x, h_y, 0, 1, 0.5)
        self._add_quadrature_coupling(h_x, h_y, 2, 3, 1.0 / np.sqrt(2.0))

        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def initial_state(self) -> qutip.Qobj:
        return qutip.Qobj(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.complex128))

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
                [np.exp(1j * theta), 0.0, -np.exp(2j * theta), 0.0],
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
