from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip
from scipy.optimize import minimize_scalar

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class TwoPhotonOpenNoiseConfig:
    """Noise and calibration settings for the open-system two-photon CZ model."""

    intermediate_detuning_offset: float = 0.0
    common_two_photon_detuning: float = 0.0
    differential_two_photon_detuning: float = 0.0
    doppler_detuning_01: float = 0.0
    doppler_detuning_11: float = 0.0
    blockade_shift_offset: float = 0.0
    lower_amplitude_scale: float = 1.0
    upper_amplitude_scale: float = 1.0
    intermediate_decay_rate: float = 0.0
    rydberg_decay_rate: float = 0.0
    intermediate_dephasing_rate: float = 0.0
    rydberg_dephasing_rate: float = 0.0
    extra_rydberg_leakage_rate: float = 0.0
    intermediate_branch_to_qubit: float = 0.5
    rydberg_branch_to_qubit: float = 0.0


@dataclass(frozen=True)
class TwoPhotonCZOpen10DModel:
    """Open-system two-photon CZ model with an explicit loss sink.

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
    9. |loss>
    """

    species: NeutralYb171Species
    lower_rabi: float
    upper_rabi: float
    intermediate_detuning: float
    blockade_shift: float
    two_photon_detuning_01: float = 0.0
    two_photon_detuning_11: float = 0.0
    upper_leg_phase: float = 0.0
    noise: TwoPhotonOpenNoiseConfig = field(default_factory=TwoPhotonOpenNoiseConfig)

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0e", "0r", "11", "W_e", "ee", "W_r", "E_er", "rr", "loss")

    def dimension(self) -> int:
        return 10

    def active_gate_indices(self) -> tuple[int, int]:
        return 0, 3

    def loss_index(self) -> int:
        return 9

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((10, 10), dtype=np.complex128)

        delta = self.intermediate_detuning + self.noise.intermediate_detuning_offset
        det_01 = (
            self.two_photon_detuning_01
            + self.noise.common_two_photon_detuning
            + self.noise.doppler_detuning_01
        )
        det_11 = (
            self.two_photon_detuning_11
            + self.noise.common_two_photon_detuning
            + self.noise.differential_two_photon_detuning
            + self.noise.doppler_detuning_11
        )
        blockade = self.blockade_shift + self.noise.blockade_shift_offset

        h_d[1, 1] = -delta
        h_d[2, 2] = -det_01
        h_d[4, 4] = -delta
        h_d[5, 5] = -2.0 * delta
        h_d[6, 6] = -det_11
        h_d[7, 7] = -(delta + det_11)
        h_d[8, 8] = blockade - 2.0 * det_11

        upper_x, upper_y = self._upper_leg_control_matrices()
        upper_scale = self.noise.upper_amplitude_scale * self.upper_rabi
        h_d += upper_scale * (
            np.cos(self.upper_leg_phase) * upper_x + np.sin(self.upper_leg_phase) * upper_y
        )
        return qutip.Qobj(h_d)

    def lower_leg_control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        lower_x = np.zeros((10, 10), dtype=np.complex128)
        lower_y = np.zeros((10, 10), dtype=np.complex128)

        self._add_quadrature_coupling(lower_x, lower_y, 0, 1, 0.5)
        self._add_quadrature_coupling(lower_x, lower_y, 3, 4, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(lower_x, lower_y, 4, 5, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(lower_x, lower_y, 6, 7, 0.5)

        return qutip.Qobj(lower_x), qutip.Qobj(lower_y)

    def control_amplitude_bound(self) -> float:
        return float(self.lower_rabi * self.noise.lower_amplitude_scale)

    def collapse_operators(self) -> list[qutip.Qobj]:
        c_ops: list[qutip.Qobj] = []
        beta_e = float(np.clip(self.noise.intermediate_branch_to_qubit, 0.0, 1.0))
        beta_r = float(np.clip(self.noise.rydberg_branch_to_qubit, 0.0, 1.0))
        gamma_e = max(self.noise.intermediate_decay_rate, 0.0)
        gamma_r = max(self.noise.rydberg_decay_rate, 0.0)
        gamma_e_phi = max(self.noise.intermediate_dephasing_rate, 0.0)
        gamma_r_phi = max(self.noise.rydberg_dephasing_rate, 0.0)
        gamma_leak = max(self.noise.extra_rydberg_leakage_rate, 0.0)
        loss = self.loss_index()

        if gamma_e > 0.0:
            self._append_jump(c_ops, gamma_e * beta_e, 0, 1)
            self._append_jump(c_ops, gamma_e * (1.0 - beta_e), loss, 1)
            self._append_jump(c_ops, gamma_e * beta_e, 3, 4)
            self._append_jump(c_ops, gamma_e * (1.0 - beta_e), loss, 4)
            self._append_jump(c_ops, 2.0 * gamma_e * beta_e, 4, 5)
            self._append_jump(c_ops, 2.0 * gamma_e * (1.0 - beta_e), loss, 5)
            self._append_jump(c_ops, gamma_e * beta_e, 6, 7)
            self._append_jump(c_ops, gamma_e * (1.0 - beta_e), loss, 7)

        if gamma_r > 0.0:
            self._append_jump(c_ops, gamma_r * beta_r, 0, 2)
            self._append_jump(c_ops, gamma_r * (1.0 - beta_r), loss, 2)
            self._append_jump(c_ops, gamma_r * beta_r, 3, 6)
            self._append_jump(c_ops, gamma_r * (1.0 - beta_r), loss, 6)
            self._append_jump(c_ops, gamma_r * beta_r, 4, 7)
            self._append_jump(c_ops, gamma_r * (1.0 - beta_r), loss, 7)
            self._append_jump(c_ops, 2.0 * gamma_r * beta_r, 6, 8)
            self._append_jump(c_ops, 2.0 * gamma_r * (1.0 - beta_r), loss, 8)

        if gamma_e_phi > 0.0:
            n_e = np.diag([0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0])
            c_ops.append(np.sqrt(gamma_e_phi) * qutip.Qobj(n_e))

        if gamma_r_phi > 0.0:
            n_r = np.diag([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
            c_ops.append(np.sqrt(gamma_r_phi) * qutip.Qobj(n_r))

        if gamma_leak > 0.0:
            self._append_jump(c_ops, gamma_leak, loss, 2)
            self._append_jump(c_ops, gamma_leak, loss, 6)
            self._append_jump(c_ops, gamma_leak, loss, 7)
            self._append_jump(c_ops, 2.0 * gamma_leak, loss, 8)

        return c_ops

    def drift_liouvillian(self) -> qutip.Qobj:
        return qutip.liouvillian(self.drift_hamiltonian(), self.collapse_operators())

    def control_liouvillians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x, h_y = self.lower_leg_control_hamiltonians()
        l_x = -1j * (qutip.spre(h_x) - qutip.spost(h_x))
        l_y = -1j * (qutip.spre(h_y) - qutip.spost(h_y))
        return l_x, l_y

    def initial_superoperator(self) -> qutip.Qobj:
        return qutip.to_super(qutip.qeye(self.dimension()))

    def target_unitary(self, theta: float = 0.0) -> qutip.Qobj:
        diagonal = np.ones(self.dimension(), dtype=np.complex128)
        diagonal[0] = np.exp(1j * theta)
        diagonal[3] = -np.exp(2j * theta)
        return qutip.Qobj(np.diag(diagonal))

    def target_superoperator(self, theta: float = 0.0) -> qutip.Qobj:
        return qutip.to_super(self.target_unitary(theta))

    def special_phase_gate_state(self) -> qutip.Qobj:
        vector = np.zeros((self.dimension(), 1), dtype=np.complex128)
        vector[self.active_gate_indices()[0], 0] = 1.0
        vector[self.active_gate_indices()[1], 0] = 1.0
        return qutip.Qobj(vector)

    def phase_gate_fidelity_from_ket(self, state: np.ndarray, theta: float) -> float:
        alpha = complex(state[self.active_gate_indices()[0]])
        beta = complex(state[self.active_gate_indices()[1]])
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
        target_01 = self._ket((unitary @ np.array([[1.0], [0.0]], dtype=np.complex128)).ravel().tolist())
        target_11 = self._ket((unitary @ np.array([[0.0], [1.0]], dtype=np.complex128)).ravel().tolist())
        target_plus = self._ket(
            (unitary @ np.array([[1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0)]], dtype=np.complex128))
            .ravel()
            .tolist()
        )
        target_plus_i = self._ket(
            (unitary @ np.array([[1.0 / np.sqrt(2.0)], [1j / np.sqrt(2.0)]], dtype=np.complex128))
            .ravel()
            .tolist()
        )
        return [
            (ket_01, target_01),
            (ket_11, target_11),
            (ket_plus, target_plus),
            (ket_plus_i, target_plus_i),
        ]

    def projector_onto_active_subspace(self) -> qutip.Qobj:
        projector = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        for index in self.active_gate_indices():
            projector[index, index] = 1.0
        return qutip.Qobj(projector)

    def probe_fidelity(self, final_states: list[qutip.Qobj], theta: float) -> float:
        probes = self.probe_kets(theta)
        projector = self.projector_onto_active_subspace()
        fidelities: list[float] = []
        for final_state, (_, target_ket) in zip(final_states, probes):
            projected = projector * final_state * projector
            fidelities.append(float(qutip.expect(qutip.ket2dm(target_ket), projected).real))
        return float(np.mean(fidelities))

    def optimize_theta_for_probe_states(self, final_states: list[qutip.Qobj]) -> tuple[float, float]:
        result = minimize_scalar(
            lambda theta: -self.probe_fidelity(final_states, theta),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        theta = float(np.mod(result.x, 2.0 * np.pi))
        return theta, self.probe_fidelity(final_states, theta)

    def evolution_hamiltonian_terms(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
        dt: float,
    ) -> list[qutip.Qobj | list[object]]:
        h_d = self.drift_hamiltonian()
        h_x, h_y = self.lower_leg_control_hamiltonians()

        def coeff_x(t: float, args: dict[str, object]) -> float:
            index = min(int(t / dt), len(ctrl_x) - 1)
            return float(ctrl_x[index])

        def coeff_y(t: float, args: dict[str, object]) -> float:
            index = min(int(t / dt), len(ctrl_y) - 1)
            return float(ctrl_y[index])

        return [h_d, [h_x, coeff_x], [h_y, coeff_y]]

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
        vector[3, 0] = active_amplitudes[1]
        return qutip.Qobj(vector)

    @staticmethod
    def _append_jump(
        c_ops: list[qutip.Qobj],
        rate: float,
        to_index: int,
        from_index: int,
    ) -> None:
        if rate <= 0.0:
            return
        operator = np.zeros((10, 10), dtype=np.complex128)
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

    def _upper_leg_control_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        upper_x = np.zeros((10, 10), dtype=np.complex128)
        upper_y = np.zeros((10, 10), dtype=np.complex128)
        self._add_quadrature_coupling(upper_x, upper_y, 1, 2, 0.5)
        self._add_quadrature_coupling(upper_x, upper_y, 4, 6, 0.5)
        self._add_quadrature_coupling(upper_x, upper_y, 5, 7, 1.0 / np.sqrt(2.0))
        self._add_quadrature_coupling(upper_x, upper_y, 7, 8, 1.0 / np.sqrt(2.0))
        return upper_x, upper_y
