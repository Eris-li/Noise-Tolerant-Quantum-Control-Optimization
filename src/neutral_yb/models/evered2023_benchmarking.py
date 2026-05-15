from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Evered2023ExponentialBenchmarkResult:
    """Synthetic repeated-gate benchmark result for an Evered-style CZ map.

    Evered et al. extract the reported experimental CZ fidelity by fitting
    repeated-gate data to an exponential decay. This helper applies the same
    fitting convention to a simulated diagonal CZ map. It is not a replacement
    for the full experimental sequence with random rotations, SPAM correction,
    atom loss handling, and shot noise; it is the repository's deterministic
    bridge from a simulated channel to the paper's per-gate decay convention.
    By default the helper assumes the coherent error has been twirled by the
    random rotations used in the benchmark, so the fitted per-gate fidelity is
    the single-gate average fidelity of the simulated map.
    """

    gate_counts: tuple[int, ...]
    fitted_fidelities: tuple[float, ...]
    fitted_amplitude: float
    gate_fidelity: float

    @property
    def gate_error(self) -> float:
        return float(1.0 - self.gate_fidelity)

    def to_json(self) -> dict[str, float | list[int] | list[float] | str]:
        return {
            "metric": "evered2023_exponential_decay_per_cz",
            "fit_model": "fidelity(N)=A*F_CZ**N, no offset",
            "default_assumption": "coherent errors randomized/twirled as in repeated-gate benchmarking",
            "gate_counts": [int(value) for value in self.gate_counts],
            "fitted_fidelities": [float(value) for value in self.fitted_fidelities],
            "fitted_amplitude": float(self.fitted_amplitude),
            "gate_fidelity": float(self.gate_fidelity),
            "gate_error": float(self.gate_error),
        }


def diagonal_cz_average_gate_fidelity(
    alpha: complex,
    beta: complex,
    theta: float,
) -> float:
    """Return the 4D computational average fidelity of a diagonal CZ map.

    The simulated symmetry-reduced models propagate the non-normalized state
    ``|01> + |11>``. If the dynamics do not mix the one-excitation and
    two-excitation computational branches, the returned amplitudes define the
    active part of the full computational map

        diag(1, alpha, alpha, beta),

    while population outside the computational basis is treated as leakage. The
    target gate is ``diag(1, exp(i theta), exp(i theta), -exp(2i theta))``.
    """

    alpha = complex(alpha)
    beta = complex(beta)
    theta = float(theta)
    phased_sum = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
    population_sum = 1.0 + 2.0 * abs(alpha) ** 2 + abs(beta) ** 2
    return float((abs(phased_sum) ** 2 + population_sum) / 20.0)


def diagonal_cz_process_fidelity(
    alpha: complex,
    beta: complex,
    theta: float,
) -> float:
    """Return the 4D computational process fidelity of a diagonal CZ map.

    For the symmetry-reduced state ``|01> + |11>``, the active computational
    map is ``diag(1, alpha, alpha, beta)`` and the target is
    ``diag(1, exp(i theta), exp(i theta), -exp(2i theta))``. This is the
    normalized process overlap ``|Tr(U_target^dagger M)|^2 / d^2`` with d=4.
    Population outside the computational subspace reduces the process overlap
    through the reduced computational amplitudes alpha and beta.
    """

    alpha = complex(alpha)
    beta = complex(beta)
    theta = float(theta)
    phased_sum = 1.0 + 2.0 * np.exp(-1j * theta) * alpha - np.exp(-2j * theta) * beta
    return float(abs(phased_sum) ** 2 / 16.0)


def repeated_diagonal_cz_average_fidelities(
    alpha: complex,
    beta: complex,
    theta: float,
    gate_counts: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    """Average gate fidelities after repeated applications of the same CZ map."""

    alpha = complex(alpha)
    beta = complex(beta)
    theta = float(theta)
    counts = np.asarray(gate_counts, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("gate_counts must be non-negative")

    fidelities = np.empty(counts.shape, dtype=np.float64)
    for index, count in np.ndenumerate(counts):
        n = int(count)
        if n == 0:
            fidelities[index] = 1.0
            continue
        alpha_n = alpha**n
        beta_n = beta**n
        phase_01 = np.exp(-1j * n * theta)
        phase_11 = ((-1.0) ** n) * np.exp(-2j * n * theta)
        phased_sum = 1.0 + 2.0 * phase_01 * alpha_n + phase_11 * beta_n
        population_sum = 1.0 + 2.0 * abs(alpha_n) ** 2 + abs(beta_n) ** 2
        fidelities[index] = float((abs(phased_sum) ** 2 + population_sum) / 20.0)
    return fidelities


def fit_exponential_decay_fidelity(
    gate_counts: tuple[int, ...] | list[int] | np.ndarray,
    fidelities: tuple[float, ...] | list[float] | np.ndarray,
) -> Evered2023ExponentialBenchmarkResult:
    """Fit repeated-gate data to ``A * F_CZ**N`` without an offset."""

    counts = np.asarray(gate_counts, dtype=np.float64)
    values = np.asarray(fidelities, dtype=np.float64)
    if counts.shape != values.shape:
        raise ValueError("gate_counts and fidelities must have the same shape")
    if counts.size < 2:
        raise ValueError("at least two benchmark points are required")
    if np.any(counts < 0.0):
        raise ValueError("gate_counts must be non-negative")
    if np.any(values <= 0.0):
        raise ValueError("fidelities must be positive for log-linear exponential fitting")

    design = np.column_stack([np.ones_like(counts), counts])
    log_amplitude, log_gate_fidelity = np.linalg.lstsq(design, np.log(values), rcond=None)[0]
    amplitude = float(np.exp(log_amplitude))
    gate_fidelity = float(np.exp(log_gate_fidelity))
    return Evered2023ExponentialBenchmarkResult(
        gate_counts=tuple(int(value) for value in counts),
        fitted_fidelities=tuple(float(value) for value in values),
        fitted_amplitude=amplitude,
        gate_fidelity=gate_fidelity,
    )


def evered2023_exponential_decay_fidelity_from_diagonal_map(
    alpha: complex,
    beta: complex,
    theta: float,
    gate_counts: tuple[int, ...] | list[int] | np.ndarray = tuple(range(0, 21, 2)),
    coherent_repetition: bool = False,
) -> Evered2023ExponentialBenchmarkResult:
    """Return the paper-style repeated-gate decay fidelity for a CZ map.

    ``coherent_repetition=False`` is the Evered-style randomized benchmark
    convention: coherent errors are assumed to be randomized/twirled, producing
    a simple exponential decay with the single-gate average fidelity as the
    decay base. ``coherent_repetition=True`` instead applies the same coherent
    diagonal map repeatedly before fitting; this is useful as a diagnostic but
    can oscillate and should not be interpreted as the paper's benchmark
    protocol.
    """

    if coherent_repetition:
        fidelities = repeated_diagonal_cz_average_fidelities(alpha, beta, theta, gate_counts)
    else:
        single_gate_fidelity = diagonal_cz_average_gate_fidelity(alpha, beta, theta)
        counts = np.asarray(gate_counts, dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("gate_counts must be non-negative")
        fidelities = single_gate_fidelity ** counts
    return fit_exponential_decay_fidelity(gate_counts, fidelities)
