from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip
from scipy.optimize import minimize_scalar

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class Ma2023NoiseConfig:
    """Noise terms for the Ma et al. 2023 metastable-Yb Rydberg gate.

    Frequencies and rates are dimensionless in units of the reference angular
    Rabi frequency. Quasistatic terms should be sampled into this object before
    constructing ensemble members.
    """

    common_detuning: float = 0.0
    differential_detuning: float = 0.0
    blockade_shift_offset: float = 0.0
    rabi_amplitude_scale: float = 1.0
    metastable_loss_rate: float = 0.0
    rydberg_decay_rate: float = 0.0
    rydberg_dephasing_rate: float = 0.0
    off_resonant_leakage_rate: float = 0.0


@dataclass(frozen=True)
class Ma2023TimeOptimal2QModel:
    """Reduced open-system model for Nature 2023 time-optimal two-qubit gates.

    Basis ordering:
    0. |01>
    1. |0r>
    2. |11>
    3. |W_r> = (|1r> + |r1>) / sqrt(2)
    4. |rr>
    5. |leak>
    6. |loss>

    The coherent block is the finite-blockade extension of the ideal global-CZ
    model. The extra sink states separate detectable erasure/loss from leakage
    inside non-computational metastable/Rydberg manifolds.
    """

    species: NeutralYb171Species
    rabi_frequency: float = 1.0
    blockade_shift: float = 16.0
    static_detuning_01: float = 0.0
    static_detuning_11: float = 0.0
    noise: Ma2023NoiseConfig = field(default_factory=Ma2023NoiseConfig)

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0r", "11", "W_r", "rr", "leak", "loss")

    def dimension(self) -> int:
        return 7

    def active_gate_indices(self) -> tuple[int, int]:
        return 0, 2

    def leak_index(self) -> int:
        return 5

    def loss_index(self) -> int:
        return 6

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        detuning_01 = self.static_detuning_01 + self.noise.common_detuning
        detuning_11 = self.static_detuning_11 + self.noise.common_detuning + self.noise.differential_detuning
        blockade = self.blockade_shift + self.noise.blockade_shift_offset

        h_d[1, 1] = -detuning_01
        h_d[3, 3] = -detuning_11
        h_d[4, 4] = blockade - 2.0 * detuning_11
        return qutip.Qobj(h_d)

    def lower_leg_control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        h_y = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        scale = float(self.rabi_frequency * self.noise.rabi_amplitude_scale)

        self._add_quadrature_coupling(h_x, h_y, 0, 1, 0.5 * scale)
        self._add_quadrature_coupling(h_x, h_y, 2, 3, (1.0 / np.sqrt(2.0)) * scale)
        self._add_quadrature_coupling(h_x, h_y, 3, 4, (1.0 / np.sqrt(2.0)) * scale)
        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def control_amplitude_bound(self) -> float:
        return float(self.rabi_frequency)

    def collapse_operators(self) -> list[qutip.Qobj]:
        c_ops: list[qutip.Qobj] = []
        gamma_meta = max(float(self.noise.metastable_loss_rate), 0.0)
        gamma_r = max(float(self.noise.rydberg_decay_rate), 0.0)
        gamma_phi = max(float(self.noise.rydberg_dephasing_rate), 0.0)
        gamma_leak = max(float(self.noise.off_resonant_leakage_rate), 0.0)
        leak = self.leak_index()
        loss = self.loss_index()

        if gamma_meta > 0.0:
            self._append_jump(c_ops, 2.0 * gamma_meta, loss, 0)
            self._append_jump(c_ops, gamma_meta, loss, 1)
            self._append_jump(c_ops, 2.0 * gamma_meta, loss, 2)
            self._append_jump(c_ops, gamma_meta, loss, 3)

        if gamma_r > 0.0:
            self._append_jump(c_ops, gamma_r, loss, 1)
            self._append_jump(c_ops, gamma_r, loss, 3)
            self._append_jump(c_ops, 2.0 * gamma_r, loss, 4)

        if gamma_phi > 0.0:
            n_r = np.diag([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0])
            c_ops.append(np.sqrt(gamma_phi) * qutip.Qobj(n_r))

        if gamma_leak > 0.0:
            self._append_jump(c_ops, gamma_leak, leak, 1)
            self._append_jump(c_ops, gamma_leak, leak, 3)
            self._append_jump(c_ops, 2.0 * gamma_leak, leak, 4)

        return c_ops

    def drift_liouvillian(self) -> qutip.Qobj:
        return qutip.liouvillian(self.drift_hamiltonian(), self.collapse_operators())

    def control_liouvillians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x, h_y = self.lower_leg_control_hamiltonians()
        l_x = -1j * (qutip.spre(h_x) - qutip.spost(h_x))
        l_y = -1j * (qutip.spre(h_y) - qutip.spost(h_y))
        return l_x, l_y

    def special_phase_gate_state(self) -> qutip.Qobj:
        vector = np.zeros((self.dimension(), 1), dtype=np.complex128)
        vector[0, 0] = 1.0
        vector[2, 0] = 1.0
        return qutip.Qobj(vector)

    def target_unitary(self, theta: float = 0.0) -> qutip.Qobj:
        diagonal = np.ones(self.dimension(), dtype=np.complex128)
        diagonal[0] = np.exp(1j * theta)
        diagonal[2] = -np.exp(2j * theta)
        return qutip.Qobj(np.diag(diagonal))

    def target_superoperator(self, theta: float = 0.0) -> qutip.Qobj:
        return qutip.to_super(self.target_unitary(theta))

    def phase_gate_fidelity_from_ket(self, state: np.ndarray, theta: float) -> float:
        alpha = complex(state[0])
        beta = complex(state[2])
        phased_sum = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
        population_sum = 1.0 + 2.0 * abs(alpha) ** 2 + abs(beta) ** 2
        return float((abs(phased_sum) ** 2 + population_sum) / 20.0)

    def optimize_theta_for_ket(self, state: np.ndarray) -> tuple[float, float]:
        result = minimize_scalar(
            lambda theta: -self.phase_gate_fidelity_from_ket(state, theta),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.phase_gate_fidelity_from_ket(state, theta)

    def probe_kets(self, theta: float) -> list[tuple[qutip.Qobj, qutip.Qobj]]:
        ket_01 = self._ket([1.0, 0.0])
        ket_11 = self._ket([0.0, 1.0])
        ket_plus = self._ket([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
        ket_plus_i = self._ket([1.0 / np.sqrt(2.0), 1j / np.sqrt(2.0)])
        unitary = np.diag([np.exp(1j * theta), -np.exp(2j * theta)])
        targets = [
            unitary @ np.array([[1.0], [0.0]], dtype=np.complex128),
            unitary @ np.array([[0.0], [1.0]], dtype=np.complex128),
            unitary @ np.array([[1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0)]], dtype=np.complex128),
            unitary @ np.array([[1.0 / np.sqrt(2.0)], [1j / np.sqrt(2.0)]], dtype=np.complex128),
        ]
        return [
            (ket_01, self._ket(targets[0].ravel().tolist())),
            (ket_11, self._ket(targets[1].ravel().tolist())),
            (ket_plus, self._ket(targets[2].ravel().tolist())),
            (ket_plus_i, self._ket(targets[3].ravel().tolist())),
        ]

    def projector_onto_active_subspace(self) -> qutip.Qobj:
        projector = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        for index in self.active_gate_indices():
            projector[index, index] = 1.0
        return qutip.Qobj(projector)

    def control_cartesian_to_polar(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2)
        phases = np.mod(np.arctan2(ctrl_y, ctrl_x), 2.0 * np.pi)
        return amplitudes, phases

    def _ket(self, active_amplitudes: list[complex]) -> qutip.Qobj:
        vector = np.zeros((self.dimension(), 1), dtype=np.complex128)
        vector[0, 0] = active_amplitudes[0]
        vector[2, 0] = active_amplitudes[1]
        return qutip.Qobj(vector)

    def _append_jump(
        self,
        c_ops: list[qutip.Qobj],
        rate: float,
        to_index: int,
        from_index: int,
    ) -> None:
        if rate <= 0.0:
            return
        operator = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        operator[to_index, from_index] = 1.0
        c_ops.append(np.sqrt(rate) * qutip.Qobj(operator))

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
