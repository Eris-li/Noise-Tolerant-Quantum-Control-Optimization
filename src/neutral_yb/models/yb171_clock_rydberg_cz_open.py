from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import qutip
from scipy.optimize import minimize_scalar
from scipy.linalg import expm

from neutral_yb.config.species import NeutralYb171Species


@dataclass(frozen=True)
class Yb171ClockRydbergNoiseConfig:
    """Noise settings for an effective full ^171Yb native CZ gate.

    The gate is modeled as three consecutive segments:

    1. a fixed clock-shelving pulse on ``|1> <-> |c>``,
    2. an optimized UV pulse on ``|c> <-> |r>``,
    3. a fixed clock-unshelving pulse back to the logical basis.

    Quasistatic detuning and amplitude errors are constant within one gate
    realization and can be sampled across an ensemble. Markovian decay and
    dephasing are represented with Lindblad operators.
    """

    common_clock_detuning: float = 0.0
    differential_clock_detuning: float = 0.0
    common_uv_detuning: float = 0.0
    differential_uv_detuning: float = 0.0
    blockade_shift_offset: float = 0.0
    clock_amplitude_scale: float = 1.0
    uv_amplitude_scale: float = 1.0
    clock_decay_rate: float = 0.0
    clock_dephasing_rate: float = 0.0
    rydberg_decay_rate: float = 0.0
    rydberg_dephasing_rate: float = 0.0
    neighboring_mf_leakage_rate: float = 0.0


@dataclass(frozen=True)
class Yb171ClockRydbergCZOpenModel:
    """Effective full-gate ^171Yb CZ model with explicit shelving pulses.

    Basis ordering:
    0. |01>
    1. |0c>
    2. |0r>
    3. |11>
    4. |W_c>  = (|1c> + |c1>) / sqrt(2)
    5. |cc>
    6. |W_r>  = (|1r> + |r1>) / sqrt(2)
    7. |W_cr> = (|cr> + |rc>) / sqrt(2)
    8. |rr>
    9. |leak>
    10. |loss>

    The active logical gate subspace remains {|01>, |11>}. The fixed prefix
    and suffix pulses model the experimental clock shelving and unshelving,
    while the optimized middle segment models the UV entangling pulse.
    """

    species: NeutralYb171Species
    uv_rabi: float
    blockade_shift: float
    clock_pi_time: float
    clock_num_steps: int = 64
    static_clock_detuning_01: float = 0.0
    static_clock_detuning_11: float = 0.0
    static_uv_detuning_01: float = 0.0
    static_uv_detuning_11: float = 0.0
    noise: Yb171ClockRydbergNoiseConfig = field(default_factory=Yb171ClockRydbergNoiseConfig)

    def basis_labels(self) -> tuple[str, ...]:
        return ("01", "0c", "0r", "11", "W_c", "cc", "W_r", "W_cr", "rr", "leak", "loss")

    def dimension(self) -> int:
        return 11

    def active_gate_indices(self) -> tuple[int, int]:
        return 0, 3

    def leak_index(self) -> int:
        return 9

    def loss_index(self) -> int:
        return 10

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        det_clock_01 = self.static_clock_detuning_01 + self.noise.common_clock_detuning
        det_clock_11 = (
            self.static_clock_detuning_11
            + self.noise.common_clock_detuning
            + self.noise.differential_clock_detuning
        )
        det_uv_01 = self.static_uv_detuning_01 + self.noise.common_uv_detuning
        det_uv_11 = self.static_uv_detuning_11 + self.noise.common_uv_detuning + self.noise.differential_uv_detuning
        blockade = self.blockade_shift + self.noise.blockade_shift_offset

        h_d[1, 1] = -det_clock_01
        h_d[2, 2] = -(det_clock_01 + det_uv_01)
        h_d[4, 4] = -det_clock_11
        h_d[5, 5] = -2.0 * det_clock_11
        h_d[6, 6] = -(det_clock_11 + det_uv_11)
        h_d[7, 7] = -(2.0 * det_clock_11 + det_uv_11)
        h_d[8, 8] = blockade - 2.0 * (det_clock_11 + det_uv_11)
        return qutip.Qobj(h_d)

    def lower_leg_control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        """UV |c> <-> |r> quadratures optimized by GRAPE."""
        h_x = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        h_y = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)

        scale = float(self.noise.uv_amplitude_scale)
        self._add_quadrature_coupling(h_x, h_y, 1, 2, 0.5 * scale)
        self._add_quadrature_coupling(h_x, h_y, 4, 6, 0.5 * scale)
        self._add_quadrature_coupling(h_x, h_y, 5, 7, (1.0 / np.sqrt(2.0)) * scale)
        self._add_quadrature_coupling(h_x, h_y, 7, 8, (1.0 / np.sqrt(2.0)) * scale)
        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def clock_control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        """Fixed clock |1> <-> |c> shelving pulse quadratures."""
        h_x = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        h_y = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)

        scale = float(self.noise.clock_amplitude_scale)
        self._add_quadrature_coupling(h_x, h_y, 0, 1, 0.5 * scale)
        self._add_quadrature_coupling(h_x, h_y, 3, 4, (1.0 / np.sqrt(2.0)) * scale)
        self._add_quadrature_coupling(h_x, h_y, 4, 5, (1.0 / np.sqrt(2.0)) * scale)
        self._add_quadrature_coupling(h_x, h_y, 6, 7, 0.5 * scale)
        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def clock_segment_controls(self) -> dict[str, np.ndarray | float]:
        envelope = np.blackman(self.clock_num_steps).astype(np.float64)
        if np.allclose(envelope, 0.0):
            envelope = np.ones(self.clock_num_steps, dtype=np.float64)
        envelope = np.clip(envelope, 0.0, None)
        dt = self.clock_pi_time / float(self.clock_num_steps)
        area = dt * float(np.sum(envelope))
        amplitude = 0.0 if area <= 0.0 else float(np.pi / area)
        ctrl_x = amplitude * envelope
        ctrl_y = np.zeros_like(ctrl_x)
        return {
            "prefix_x": ctrl_x,
            "prefix_y": ctrl_y,
            "prefix_dt": float(dt),
            "suffix_x": ctrl_x.copy(),
            "suffix_y": ctrl_y.copy(),
            "suffix_dt": float(dt),
        }

    @cached_property
    def fixed_clock_segment_cache(self) -> dict[str, object]:
        h_d = np.asarray(self.drift_hamiltonian().full(), dtype=np.complex128)
        h_clock_x, h_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in self.clock_control_hamiltonians()
        ]
        decay_matrix = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        for operator in self.collapse_operators():
            c_matrix = np.asarray(operator.full(), dtype=np.complex128)
            decay_matrix += c_matrix.conj().T @ c_matrix
        g_d = -1j * h_d - 0.5 * decay_matrix
        g_clock_x = -1j * h_clock_x
        g_clock_y = -1j * h_clock_y
        l_d = np.asarray(self.drift_liouvillian().full(), dtype=np.complex128)
        l_clock_x, l_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in self.clock_control_liouvillians()
        ]
        segments = self.clock_segment_controls()

        def build_phase_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(dt * (g_d + float(x_value) * g_clock_x + float(y_value) * g_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        def build_liou_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(dt * (l_d + float(x_value) * l_clock_x + float(y_value) * l_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        prefix_x = np.asarray(segments["prefix_x"], dtype=np.float64)
        prefix_y = np.asarray(segments["prefix_y"], dtype=np.float64)
        suffix_x = np.asarray(segments["suffix_x"], dtype=np.float64)
        suffix_y = np.asarray(segments["suffix_y"], dtype=np.float64)
        prefix_dt = float(segments["prefix_dt"])
        suffix_dt = float(segments["suffix_dt"])
        phase_identity = np.eye(self.dimension(), dtype=np.complex128)
        liou_identity = np.eye(self.dimension() * self.dimension(), dtype=np.complex128)

        prefix_phase_steps = build_phase_steps(prefix_x, prefix_y, prefix_dt)
        suffix_phase_steps = build_phase_steps(suffix_x, suffix_y, suffix_dt)
        prefix_liou_steps = build_liou_steps(prefix_x, prefix_y, prefix_dt)
        suffix_liou_steps = build_liou_steps(suffix_x, suffix_y, suffix_dt)

        return {
            "phase_prefix": self._compose_propagators(prefix_phase_steps, phase_identity),
            "phase_suffix": self._compose_propagators(suffix_phase_steps, phase_identity),
            "liou_prefix": self._compose_propagators(prefix_liou_steps, liou_identity),
            "liou_suffix": self._compose_propagators(suffix_liou_steps, liou_identity),
        }

    @cached_property
    def fixed_clock_trajectory_cache(self) -> dict[str, object]:
        h_d = np.asarray(self.drift_hamiltonian().full(), dtype=np.complex128)
        l_d = np.asarray(self.drift_liouvillian().full(), dtype=np.complex128)
        h_clock_x, h_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in self.clock_control_hamiltonians()
        ]
        l_clock_x, l_clock_y = [
            np.asarray(operator.full(), dtype=np.complex128) for operator in self.clock_control_liouvillians()
        ]
        segments = self.clock_segment_controls()

        def build_liou_steps(ctrl_x: np.ndarray, ctrl_y: np.ndarray, dt: float) -> list[np.ndarray]:
            return [
                expm(dt * (l_d + float(x_value) * l_clock_x + float(y_value) * l_clock_y))
                for x_value, y_value in zip(ctrl_x, ctrl_y)
            ]

        prefix_x = np.asarray(segments["prefix_x"], dtype=np.float64)
        prefix_y = np.asarray(segments["prefix_y"], dtype=np.float64)
        suffix_x = np.asarray(segments["suffix_x"], dtype=np.float64)
        suffix_y = np.asarray(segments["suffix_y"], dtype=np.float64)
        prefix_dt = float(segments["prefix_dt"])
        suffix_dt = float(segments["suffix_dt"])

        prefix_liou_steps = build_liou_steps(prefix_x, prefix_y, prefix_dt)
        suffix_liou_steps = build_liou_steps(suffix_x, suffix_y, suffix_dt)
        _ = h_d  # keep parity with fixed segment construction and avoid stale imports

        return {
            "liou_prefix_steps": prefix_liou_steps,
            "liou_prefix_dts": [prefix_dt] * len(prefix_liou_steps),
            "liou_suffix_steps": suffix_liou_steps,
            "liou_suffix_dts": [suffix_dt] * len(suffix_liou_steps),
        }

    def fixed_prefix_duration(self) -> float:
        return float(self.clock_pi_time)

    def fixed_suffix_duration(self) -> float:
        return float(self.clock_pi_time)

    def total_gate_time(self, uv_segment_time: float) -> float:
        return float(self.clock_pi_time + uv_segment_time + self.clock_pi_time)

    def control_amplitude_bound(self) -> float:
        return float(self.uv_rabi)

    def collapse_operators(self) -> list[qutip.Qobj]:
        c_ops: list[qutip.Qobj] = []
        gamma_c = max(float(self.noise.clock_decay_rate), 0.0)
        gamma_c_phi = max(float(self.noise.clock_dephasing_rate), 0.0)
        gamma_r = max(float(self.noise.rydberg_decay_rate), 0.0)
        gamma_r_phi = max(float(self.noise.rydberg_dephasing_rate), 0.0)
        gamma_leak = max(float(self.noise.neighboring_mf_leakage_rate), 0.0)
        leak = self.leak_index()
        loss = self.loss_index()

        if gamma_c > 0.0:
            self._append_jump(c_ops, gamma_c, loss, 1)
            self._append_jump(c_ops, gamma_c, loss, 4)
            self._append_jump(c_ops, 2.0 * gamma_c, loss, 5)
            self._append_jump(c_ops, gamma_c, loss, 7)

        if gamma_r > 0.0:
            self._append_jump(c_ops, gamma_r, loss, 2)
            self._append_jump(c_ops, gamma_r, loss, 6)
            self._append_jump(c_ops, gamma_r, loss, 7)
            self._append_jump(c_ops, 2.0 * gamma_r, loss, 8)

        if gamma_c_phi > 0.0:
            n_c = np.diag([0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            c_ops.append(np.sqrt(gamma_c_phi) * qutip.Qobj(n_c))

        if gamma_r_phi > 0.0:
            n_r = np.diag([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0])
            c_ops.append(np.sqrt(gamma_r_phi) * qutip.Qobj(n_r))

        if gamma_leak > 0.0:
            self._append_jump(c_ops, gamma_leak, leak, 2)
            self._append_jump(c_ops, gamma_leak, leak, 6)
            self._append_jump(c_ops, gamma_leak, leak, 7)
            self._append_jump(c_ops, 2.0 * gamma_leak, leak, 8)

        return c_ops

    def drift_liouvillian(self) -> qutip.Qobj:
        return qutip.liouvillian(self.drift_hamiltonian(), self.collapse_operators())

    def control_liouvillians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x, h_y = self.lower_leg_control_hamiltonians()
        l_x = -1j * (qutip.spre(h_x) - qutip.spost(h_x))
        l_y = -1j * (qutip.spre(h_y) - qutip.spost(h_y))
        return l_x, l_y

    def clock_control_liouvillians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x, h_y = self.clock_control_hamiltonians()
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
        for final_state, (_source, target_ket) in zip(final_states, probes):
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
        vector[self.active_gate_indices()[0], 0] = active_amplitudes[0]
        vector[self.active_gate_indices()[1], 0] = active_amplitudes[1]
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
    def _compose_propagators(propagators: list[np.ndarray], identity: np.ndarray) -> np.ndarray:
        total = np.array(identity, copy=True)
        for propagator in propagators:
            total = propagator @ total
        return total

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
