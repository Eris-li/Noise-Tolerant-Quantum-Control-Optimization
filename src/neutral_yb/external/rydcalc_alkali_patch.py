from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import constants as cs


@dataclass(frozen=True)
class RydcalcEnergyPatch:
    n: int
    l: int
    j: float
    old_energy_hz: float
    new_energy_hz: float
    old_nu: float | None
    new_nu: float


def _effective_nu(atom: object, energy_hz: float) -> float:
    rydberg_hz = cs.physical_constants["Rydberg constant times c in Hz"][0]
    reduced_rydberg_hz = rydberg_hz * float(getattr(atom, "mu", 1.0))
    ionization_hz = float(getattr(getattr(atom, "core"), "Ei_Hz", 0.0))
    binding_hz = ionization_hz - float(energy_hz)
    if binding_hz <= 0.0:
        raise ValueError(f"state is not bound relative to ionization limit: {energy_hz=}")
    return float(1.0 / np.sqrt(binding_hz / reduced_rydberg_hz))


def patch_state_energy_from_arc(
    state: object,
    arc_atom: object,
    *,
    n: int,
    l: int,
    j: float,
) -> RydcalcEnergyPatch:
    """Patch one rydcalc alkali state with an ARC/NIST anchored energy.

    The local rydcalc alkali models construct states from Rydberg-Ritz quantum
    defects. That is appropriate for high-n Rydberg states but can fail badly
    for low-lying states such as Cs 6S. ARC keeps NIST low-state energies and
    falls back to quantum defects only above its species-specific cutoff. This
    helper preserves the rydcalc state object and wavefunction machinery while
    replacing the low-state energy with ARC's value.
    """

    old_energy_hz = float(getattr(state, "energy_Hz"))
    old_nu = getattr(state, "nu", None)
    old_nu_float = None if old_nu is None else float(old_nu)

    new_energy_hz = float(arc_atom.getEnergy(n, l, j) * cs.e / cs.h)
    new_nu = _effective_nu(getattr(state, "atom"), new_energy_hz)

    setattr(state, "energy_Hz", new_energy_hz)
    setattr(state, "nu", new_nu)
    setattr(state, "nub", new_nu)

    # rydcalc memoizes Numerov grids and interpolators directly on the state.
    # If the energy is changed after a matrix-element call, none of those
    # wavefunction objects may be reused.
    for name in ("wf_x", "wf_y", "wf_interp", "wf_x_min", "wf_x_max"):
        if hasattr(state, name):
            delattr(state, name)

    return RydcalcEnergyPatch(
        n=int(n),
        l=int(l),
        j=float(j),
        old_energy_hz=old_energy_hz,
        new_energy_hz=new_energy_hz,
        old_nu=old_nu_float,
        new_nu=new_nu,
    )
