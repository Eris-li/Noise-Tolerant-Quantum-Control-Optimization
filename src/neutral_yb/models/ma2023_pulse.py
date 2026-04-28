from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ma2023GaussianEdgePulse:
    """Phase-only pulse family used for the Ma 2023 method-first gate.

    The amplitude envelope is fixed by a total duration plus Gaussian rising
    and falling edges. The optimizer should vary only the optical phase.
    """

    num_tslots: int
    edge_fraction: float = 0.20
    sigma_fraction: float = 0.08

    def envelope(self) -> np.ndarray:
        return gaussian_edge_envelope(
            self.num_tslots,
            self.edge_fraction,
            self.sigma_fraction,
        )


def gaussian_edge_envelope(
    num_tslots: int,
    edge_fraction: float,
    sigma_fraction: float,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(num_tslots), endpoint=True, dtype=np.float64)
    edge = float(np.clip(edge_fraction, 1e-6, 0.499))
    sigma = float(np.clip(sigma_fraction, 1e-6, edge))
    envelope = np.ones_like(grid)
    left = grid < edge
    right = grid > 1.0 - edge
    edge_floor = np.exp(-0.5 * (edge / sigma) ** 2)
    if np.any(left):
        left_gaussian = np.exp(-0.5 * ((grid[left] - edge) / sigma) ** 2)
        envelope[left] = (left_gaussian - edge_floor) / max(1.0 - edge_floor, 1e-12)
    if np.any(right):
        right_gaussian = np.exp(-0.5 * ((grid[right] - (1.0 - edge)) / sigma) ** 2)
        envelope[right] = (right_gaussian - edge_floor) / max(1.0 - edge_floor, 1e-12)
    envelope[0] = 0.0
    envelope[-1] = 0.0
    return np.clip(envelope, 0.0, 1.0).astype(np.float64)


def wrap_phase(phases: np.ndarray) -> np.ndarray:
    return (np.asarray(phases, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_phase_difference(values: np.ndarray) -> np.ndarray:
    return wrap_phase(values)


def controls_from_envelope_phase(
    envelope: np.ndarray,
    phases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bounded_phases = wrap_phase(phases)
    amplitudes = np.asarray(envelope, dtype=np.float64)
    return (
        (amplitudes * np.cos(bounded_phases)).astype(np.float64),
        (amplitudes * np.sin(bounded_phases)).astype(np.float64),
    )


def phase_regularization(
    phases: np.ndarray,
    smoothness_weight: float,
    curvature_weight: float,
) -> tuple[float, np.ndarray]:
    phases = np.asarray(phases, dtype=np.float64)
    cost = 0.0
    gradient = np.zeros_like(phases)
    if phases.size >= 2 and smoothness_weight > 0.0:
        delta = wrap_phase_difference(phases[1:] - phases[:-1])
        smooth_cost = float(np.sum(delta**2))
        cost += smoothness_weight * smooth_cost
        gradient[:-1] -= 2.0 * smoothness_weight * delta
        gradient[1:] += 2.0 * smoothness_weight * delta
    if phases.size >= 3 and curvature_weight > 0.0:
        delta = wrap_phase_difference(phases[1:] - phases[:-1])
        curvature = wrap_phase_difference(delta[1:] - delta[:-1])
        curve_cost = float(np.sum(curvature**2))
        cost += curvature_weight * curve_cost
        gradient[:-2] += 2.0 * curvature_weight * curvature
        gradient[1:-1] -= 4.0 * curvature_weight * curvature
        gradient[2:] += 2.0 * curvature_weight * curvature
    return cost, gradient


def validate_phase_only_pulse(
    amplitudes: np.ndarray,
    phases: np.ndarray,
    *,
    tol: float = 1e-12,
) -> dict[str, bool | float]:
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    return {
        "amplitude_starts_at_zero": bool(abs(float(amplitudes[0])) <= tol),
        "amplitude_ends_at_zero": bool(abs(float(amplitudes[-1])) <= tol),
        "amplitude_within_bound": bool(float(np.max(amplitudes)) <= 1.0 + tol),
        "phase_within_minus_pi_pi": bool(
            np.all(phases >= -np.pi - tol) and np.all(phases <= np.pi + tol)
        ),
        "min_phase": float(np.min(phases)),
        "max_phase": float(np.max(phases)),
        "max_amplitude": float(np.max(amplitudes)),
    }
