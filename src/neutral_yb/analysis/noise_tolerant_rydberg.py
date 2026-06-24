from __future__ import annotations

import numpy as np
from scipy.linalg import expm, expm_frechet


BranchStates = tuple[np.ndarray, np.ndarray, np.ndarray]
BranchHamiltonians = tuple[np.ndarray, np.ndarray, np.ndarray]


def rabi_from_phase(phase: float, amplitude: float = 1.0) -> complex:
    return complex(float(amplitude) * np.exp(1j * float(phase)))


def blockade_branch_hamiltonians(
    phase: float,
    *,
    amplitude_error: float = 0.0,
    detuning: float = 0.0,
    detuning_1: float | None = None,
    detuning_2: float | None = None,
) -> BranchHamiltonians:
    omega = (1.0 + float(amplitude_error)) * rabi_from_phase(float(phase))
    delta_1 = float(detuning) if detuning_1 is None else float(detuning_1)
    delta_2 = float(detuning) if detuning_2 is None else float(detuning_2)
    delta_plus = 0.5 * (delta_1 + delta_2)
    delta_minus = 0.5 * (delta_1 - delta_2)

    h10 = np.zeros((2, 2), dtype=np.complex128)
    h01 = np.zeros((2, 2), dtype=np.complex128)
    h11 = np.zeros((3, 3), dtype=np.complex128)

    h10[0, 1] = np.conj(omega) / 2.0
    h10[1, 0] = omega / 2.0
    h10[1, 1] = delta_1

    h01[0, 1] = np.conj(omega) / 2.0
    h01[1, 0] = omega / 2.0
    h01[1, 1] = delta_2

    h11[0, 1] = np.conj(omega) / np.sqrt(2.0)
    h11[1, 0] = omega / np.sqrt(2.0)
    h11[1, 1] = delta_plus
    h11[2, 2] = delta_plus
    h11[1, 2] = delta_minus
    h11[2, 1] = delta_minus

    return h10, h01, h11


def amplitude_noise_hamiltonians(phase: float) -> BranchHamiltonians:
    return blockade_branch_hamiltonians(float(phase))


def detuning_noise_hamiltonians(
    sign: float | None = 1.0,
    *,
    detuning_1_sign: float | None = None,
    detuning_2_sign: float | None = None,
) -> BranchHamiltonians:
    if detuning_1_sign is None and detuning_2_sign is None:
        delta_1 = float(1.0 if sign is None else sign)
        delta_2 = float(1.0 if sign is None else sign)
    else:
        delta_1 = 0.0 if detuning_1_sign is None else float(detuning_1_sign)
        delta_2 = 0.0 if detuning_2_sign is None else float(detuning_2_sign)
    delta_plus = 0.5 * (delta_1 + delta_2)
    delta_minus = 0.5 * (delta_1 - delta_2)

    h10 = np.zeros((2, 2), dtype=np.complex128)
    h01 = np.zeros((2, 2), dtype=np.complex128)
    h11 = np.zeros((3, 3), dtype=np.complex128)

    h10[1, 1] = delta_1
    h01[1, 1] = delta_2
    h11[1, 1] = delta_plus
    h11[2, 2] = delta_plus
    h11[1, 2] = delta_minus
    h11[2, 1] = delta_minus

    return h10, h01, h11


def propagate_blockade_branches(
    phases: np.ndarray,
    *,
    total_time: float,
    amplitude_error: float = 0.0,
    detuning_trace: np.ndarray | None = None,
    detuning_1_trace: np.ndarray | None = None,
    detuning_2_trace: np.ndarray | None = None,
) -> BranchStates:
    phases = np.asarray(phases, dtype=np.float64)
    slots = int(phases.size)
    dt = float(total_time) / slots
    common_detuning = np.zeros(slots, dtype=np.float64) if detuning_trace is None else np.asarray(detuning_trace, dtype=np.float64)
    if common_detuning.shape != (slots,):
        raise ValueError(f"detuning_trace must have shape {(slots,)}")
    detuning_1 = common_detuning if detuning_1_trace is None else np.asarray(detuning_1_trace, dtype=np.float64)
    detuning_2 = common_detuning if detuning_2_trace is None else np.asarray(detuning_2_trace, dtype=np.float64)
    if detuning_1.shape != (slots,):
        raise ValueError(f"detuning_1_trace must have shape {(slots,)}")
    if detuning_2.shape != (slots,):
        raise ValueError(f"detuning_2_trace must have shape {(slots,)}")

    states: list[np.ndarray] = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0, 0.0], dtype=np.complex128),
    ]
    for index, phase in enumerate(phases):
        hamiltonians = blockade_branch_hamiltonians(
            float(phase),
            amplitude_error=float(amplitude_error),
            detuning_1=float(detuning_1[index]),
            detuning_2=float(detuning_2[index]),
        )
        for branch_index, hamiltonian in enumerate(hamiltonians):
            states[branch_index] = expm(-1j * hamiltonian * dt) @ states[branch_index]

    return states[0], states[1], states[2]


def propagate_first_order_sensitivity(
    phases: np.ndarray,
    *,
    total_time: float,
    noise_kind: str,
    detuning_sign_trace: np.ndarray | None = None,
    detuning_1_sign_trace: np.ndarray | None = None,
    detuning_2_sign_trace: np.ndarray | None = None,
) -> tuple[BranchStates, BranchStates]:
    phases = np.asarray(phases, dtype=np.float64)
    slots = int(phases.size)
    dt = float(total_time) / slots
    if noise_kind not in {"amplitude", "detuning"}:
        raise ValueError("noise_kind must be 'amplitude' or 'detuning'")
    common_signs = np.ones(slots, dtype=np.float64) if detuning_sign_trace is None else np.asarray(detuning_sign_trace, dtype=np.float64)
    if common_signs.shape != (slots,):
        raise ValueError(f"detuning_sign_trace must have shape {(slots,)}")
    if detuning_1_sign_trace is None and detuning_2_sign_trace is None:
        detuning_1_signs = common_signs
        detuning_2_signs = common_signs
    else:
        detuning_1_signs = (
            np.zeros(slots, dtype=np.float64)
            if detuning_1_sign_trace is None
            else np.asarray(detuning_1_sign_trace, dtype=np.float64)
        )
        detuning_2_signs = (
            np.zeros(slots, dtype=np.float64)
            if detuning_2_sign_trace is None
            else np.asarray(detuning_2_sign_trace, dtype=np.float64)
        )
    if detuning_1_signs.shape != (slots,):
        raise ValueError(f"detuning_1_sign_trace must have shape {(slots,)}")
    if detuning_2_signs.shape != (slots,):
        raise ValueError(f"detuning_2_sign_trace must have shape {(slots,)}")

    states: list[np.ndarray] = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0, 0.0], dtype=np.complex128),
    ]
    sensitivities: list[np.ndarray] = [np.zeros_like(state) for state in states]

    for index, phase in enumerate(phases):
        nominal_hamiltonians = blockade_branch_hamiltonians(float(phase))
        if noise_kind == "amplitude":
            noise_hamiltonians = amplitude_noise_hamiltonians(float(phase))
        else:
            noise_hamiltonians = detuning_noise_hamiltonians(
                detuning_1_sign=float(detuning_1_signs[index]),
                detuning_2_sign=float(detuning_2_signs[index]),
            )

        for branch_index, (h0, h1) in enumerate(zip(nominal_hamiltonians, noise_hamiltonians, strict=True)):
            generator = -1j * h0 * dt
            propagator = expm(generator)
            frechet = expm_frechet(generator, -1j * h1 * dt, compute_expm=False)
            sensitivities[branch_index] = propagator @ sensitivities[branch_index] + frechet @ states[branch_index]
            states[branch_index] = propagator @ states[branch_index]

    return (states[0], states[1], states[2]), (sensitivities[0], sensitivities[1], sensitivities[2])
