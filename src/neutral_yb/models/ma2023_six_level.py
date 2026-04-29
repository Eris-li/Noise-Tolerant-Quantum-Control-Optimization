from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip


@dataclass(frozen=True)
class Ma2023SixLevelNoiseConfig:
    """Noise and static offsets for the Ma 2023 six-level blockade model."""

    common_detuning: float = 0.0
    rydberg_zeeman_offset: float = 0.0
    rabi_amplitude_scale: float = 1.0
    rydberg_decay_rate: float = 0.0
    rydberg_decay_detected_fraction: float = 0.5
    rydberg_dephasing_rate: float = 0.0


@dataclass(frozen=True)
class Ma2023PerfectBlockadeSixLevelModel:
    """Ma 2023 Methods model with all four 6s59s 3S1 F=3/2 Rydberg sublevels.

    The single-atom basis is
    ``|0>, |1>, |r_-3/2>, |r_-1/2>, |r_1/2>, |r_3/2>``.
    In the perfect-blockade limit, double-Rydberg states are removed and the
    two-atom dynamics split into the three five-dimensional subspaces described
    in the Methods for initial states ``|00>``, ``|01>``/``|10>``, and ``|11>``.
    """

    delta_r: float = 5.8
    delta_m: float = 0.0
    rabi_frequency: float = 1.0
    include_loss_state: bool = True
    noise: Ma2023SixLevelNoiseConfig = field(default_factory=Ma2023SixLevelNoiseConfig)

    def sector_labels(self) -> tuple[str, ...]:
        return ("00", "01", "11")

    def basis_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for sector in self.sector_labels():
            labels.extend(self._sector_basis_labels(sector))
        if self.include_loss_state:
            labels.extend(("detected_decay", "undetected_decay"))
        return tuple(labels)

    def dimension(self) -> int:
        return 15 + 2 * int(self.include_loss_state)

    def computational_indices(self) -> tuple[int, int, int]:
        return 0, 5, 10

    def active_gate_indices(self) -> tuple[int, int, int]:
        return self.computational_indices()

    def loss_index(self) -> int | None:
        return 15 if self.include_loss_state else None

    def erasure_index(self) -> int | None:
        return self.detected_decay_index()

    def detected_decay_index(self) -> int | None:
        return 15 if self.include_loss_state else None

    def undetected_decay_index(self) -> int | None:
        return 16 if self.include_loss_state else None

    def loss_indices(self) -> tuple[int, ...]:
        if not self.include_loss_state:
            return ()
        return 15, 16

    def transition_subspace_indices(self, sector: str) -> tuple[int, int, int, int, int]:
        sector_index = self.sector_labels().index(sector)
        start = 5 * sector_index
        return tuple(range(start, start + 5))

    def drift_hamiltonian(self) -> qutip.Qobj:
        h_d = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        for sector in self.sector_labels():
            indices = self.transition_subspace_indices(sector)
            block = self._sector_drift_block(sector)
            h_d[np.ix_(indices, indices)] = block
        return qutip.Qobj(h_d)

    def lower_leg_control_hamiltonians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        h_y = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
        scale = float(self.rabi_frequency * self.noise.rabi_amplitude_scale)
        for sector in self.sector_labels():
            indices = self.transition_subspace_indices(sector)
            for left, right, coefficient in self._sector_couplings(sector):
                self._add_quadrature_coupling(
                    h_x,
                    h_y,
                    indices[left],
                    indices[right],
                    scale * coefficient,
                )
        return qutip.Qobj(h_x), qutip.Qobj(h_y)

    def control_amplitude_bound(self) -> float:
        return float(self.rabi_frequency)

    def collapse_operators(self) -> list[qutip.Qobj]:
        c_ops: list[qutip.Qobj] = []
        detected_decay = self.detected_decay_index()
        undetected_decay = self.undetected_decay_index()
        gamma_r = max(float(self.noise.rydberg_decay_rate), 0.0)
        detected_fraction = float(np.clip(self.noise.rydberg_decay_detected_fraction, 0.0, 1.0))
        gamma_detected = detected_fraction * gamma_r
        gamma_undetected = (1.0 - detected_fraction) * gamma_r
        gamma_phi = max(float(self.noise.rydberg_dephasing_rate), 0.0)
        if gamma_r > 0.0 and detected_decay is not None and undetected_decay is not None:
            for sector in self.sector_labels():
                for index in self.transition_subspace_indices(sector)[1:]:
                    if gamma_detected > 0.0:
                        self._append_jump(c_ops, gamma_detected, detected_decay, index)
                    if gamma_undetected > 0.0:
                        self._append_jump(c_ops, gamma_undetected, undetected_decay, index)
        if gamma_phi > 0.0:
            projector = np.zeros((self.dimension(), self.dimension()), dtype=np.complex128)
            for sector in self.sector_labels():
                for index in self.transition_subspace_indices(sector)[1:]:
                    projector[index, index] = 1.0
            c_ops.append(np.sqrt(gamma_phi) * qutip.Qobj(projector))
        return c_ops

    def drift_liouvillian(self) -> qutip.Qobj:
        return qutip.liouvillian(self.drift_hamiltonian(), self.collapse_operators())

    def control_liouvillians(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        h_x, h_y = self.lower_leg_control_hamiltonians()
        return (
            -1j * (qutip.spre(h_x) - qutip.spost(h_x)),
            -1j * (qutip.spre(h_y) - qutip.spost(h_y)),
        )

    def computational_ket(self, sector: str) -> qutip.Qobj:
        vector = np.zeros((self.dimension(), 1), dtype=np.complex128)
        vector[self.transition_subspace_indices(sector)[0], 0] = 1.0
        return qutip.Qobj(vector)

    def target_phases(self, theta0: float, theta1: float) -> tuple[complex, complex, complex]:
        return (
            np.exp(1j * theta0),
            np.exp(1j * (theta0 + theta1)),
            -np.exp(1j * (theta0 + 2.0 * theta1)),
        )

    def control_cartesian_to_polar(
        self,
        ctrl_x: np.ndarray,
        ctrl_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        amplitudes = np.sqrt(ctrl_x**2 + ctrl_y**2)
        phases = np.arctan2(ctrl_y, ctrl_x)
        return amplitudes, phases

    def _sector_drift_block(self, sector: str) -> np.ndarray:
        delta_r = float(self.delta_r + self.noise.rydberg_zeeman_offset)
        common = float(self.noise.common_detuning)
        block = np.zeros((5, 5), dtype=np.complex128)
        for local_index, rydberg_label in enumerate(self._sector_rydberg_labels(sector), start=1):
            block[local_index, local_index] = self._rydberg_detuning(rydberg_label, delta_r) - common
        return block

    def _sector_couplings(self, sector: str) -> tuple[tuple[int, int, float], ...]:
        labels = self._sector_basis_labels(sector)
        return tuple(
            (0, local_index, self._clebsch_coefficient(self._extract_rydberg_label(label)))
            for local_index, label in enumerate(labels[1:], start=1)
        )

    def _sector_basis_labels(self, sector: str) -> tuple[str, ...]:
        if sector == "00":
            return ("00", "0 r_-3/2", "0 r_1/2", "r_-3/2 0", "r_1/2 0")
        if sector == "01":
            return ("01", "0 r_-1/2", "0 r_3/2", "r_-3/2 1", "r_1/2 1")
        if sector == "11":
            return ("11", "1 r_-1/2", "1 r_3/2", "r_-1/2 1", "r_3/2 1")
        raise ValueError(f"Unsupported sector: {sector}")

    def _sector_rydberg_labels(self, sector: str) -> tuple[str, str, str, str]:
        return tuple(self._extract_rydberg_label(label) for label in self._sector_basis_labels(sector)[1:])  # type: ignore[return-value]

    @staticmethod
    def _extract_rydberg_label(two_atom_label: str) -> str:
        for token in two_atom_label.split():
            if token.startswith("r_"):
                return token
        raise ValueError(f"No Rydberg label found in {two_atom_label}")

    @staticmethod
    def _rydberg_detuning(rydberg_label: str, delta_r: float) -> float:
        return {
            "r_-3/2": -3.0 * delta_r,
            "r_-1/2": -2.0 * delta_r,
            "r_1/2": -1.0 * delta_r,
            "r_3/2": 0.0,
        }[rydberg_label]

    @staticmethod
    def _clebsch_coefficient(rydberg_label: str) -> float:
        if rydberg_label in {"r_-3/2", "r_3/2"}:
            return 0.5
        if rydberg_label in {"r_-1/2", "r_1/2"}:
            return 0.5 / np.sqrt(3.0)
        raise ValueError(f"Unsupported Rydberg label: {rydberg_label}")

    def _append_jump(
        self,
        c_ops: list[qutip.Qobj],
        rate: float,
        to_index: int,
        from_index: int,
    ) -> None:
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
