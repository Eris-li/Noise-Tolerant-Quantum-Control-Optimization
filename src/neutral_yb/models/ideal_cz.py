from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class IdealCZModel:
    """Reduced infinite-blockade model for the frozen reference experiment.

    Basis ordering:
    0. |00>
    1. |01>
    2. |0r>
    3. |10>
    4. |r0>
    5. |11>
    6. |W> = (|1r> + |r1>) / sqrt(2)
    """

    species: NeutralYb171Species

    basis_labels: Sequence[str] = ("00", "01", "0r", "10", "r0", "11", "W")
    computational_indices: Sequence[int] = (0, 1, 3, 5)

    def dimension(self) -> int:
        return len(self.basis_labels)

    def drift_hamiltonian(self) -> qutip.Qobj:
        return qutip.Qobj(np.zeros((self.dimension(), self.dimension()), dtype=np.complex128))

    def collapse_operators(self) -> tuple[qutip.Qobj, ...]:
        return ()

    def control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        h_y = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)

        self._add_quadrature_coupling(h_x, h_y, 1, 2, 0.5)
        self._add_quadrature_coupling(h_x, h_y, 3, 4, 0.5)
        self._add_quadrature_coupling(h_x, h_y, 5, 6, 1.0 / np.sqrt(2.0))

        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def full_target_unitary(self) -> qutip.Qobj:
        target = np.eye(self.dimension(), dtype=np.complex128)
        target[5, 5] = -1.0
        return qutip.Qobj(target)

    def computational_projector(self) -> np.ndarray:
        projector = np.zeros((self.dimension(), len(self.computational_indices)), dtype=np.complex128)
        for col, row in enumerate(self.computational_indices):
            projector[row, col] = 1.0
        return projector

    def computational_target(self) -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

    def best_entangling_family_phase(self, unitary: np.ndarray, num_samples: int = 2048) -> tuple[float, float]:
        reduced = self.computational_projector().conj().T @ unitary @ self.computational_projector()
        gammas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)

        best_gamma = 0.0
        best_fidelity = -np.inf
        for gamma in gammas:
            family_target = np.diag(
                [1.0, np.exp(1j * gamma), np.exp(1j * gamma), -np.exp(2j * gamma)]
            ).astype(np.complex128)
            score = abs(np.trace(family_target.conj().T @ reduced)) ** 2 / 16.0
            if score > best_fidelity:
                best_gamma = float(gamma)
                best_fidelity = float(score)

        return best_gamma, best_fidelity

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
