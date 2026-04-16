from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NeutralYb171Species:
    """Species-level configuration for neutral 171Yb.

    The reference experiment intentionally uses dimensionless units so that
    the ideal infinite-blockade model remains close to the assumptions in
    arXiv:2202.00903. The species object keeps the code semantically anchored
    to neutral 171Yb and provides a place for future physical parameters.
    """

    name: str = "171Yb"
    qubit_encoding: str = "ground_state_nuclear_spin"
    control_basis: str = "global_phase_modulated_rydberg_drive"
    omega_max: float = 1.0
    rydberg_principal_n: int = 60
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Ideal reference uses dimensionless units with omega_max = 1.",
            "Infinite blockade excludes the |rr> manifold.",
            "No noise, no detuning, and no open-system terms in the frozen benchmark.",
        )
    )


def idealised_yb171() -> NeutralYb171Species:
    return NeutralYb171Species()

