# Evered 2023 Parallel CZ Reproduction

This line is a GRAPE validation target based on the CZ gate ingredients from
Evered et al., Nature 622, 268-272 (2023), DOI `10.1038/s41586-023-06481-y`.
It is intentionally named `evered2023_parallel_cz`, parallel to
`ma2023_time_optimal_2q`, rather than as a new numbered `v*` model.

## Scope

The first implementation target is the fixed-amplitude, parameterized
time-optimal gate from Methods Eq. (1):

```math
\phi(t) = A \cos(\omega t - \varphi_0) + \delta_0 t
```

with

```math
A = 2\pi \times 0.1122,\quad
\omega = 1.0431\,\Omega,\quad
\varphi_0 = -0.7318,\quad
\delta_0 = 0,\quad
\Omega T / 2\pi = 1.215.
```

At the reported experimental scale `Omega / 2pi = 4.6 MHz`, this corresponds to
`T = 1.215 / 4.6 MHz = 264.13 ns`.

## Implemented Pieces

- Model and pulse helpers:
  [evered2023_parallel_cz.py](../src/neutral_yb/models/evered2023_parallel_cz.py)
- Reproduction entry point:
  [reproduce_evered2023_parallel_cz_gate.py](../experiments/reproduce_evered2023_parallel_cz_gate.py)
- Plotting entry point:
  [plot_evered2023_parallel_cz.py](../scripts/plot_evered2023_parallel_cz.py)
- Tests:
  [test_evered2023_parallel_cz.py](../tests/test_evered2023_parallel_cz.py)
- Artifact directory:
  `artifacts/evered2023_parallel_cz/`

The fixed-amplitude profile is not used as a GRAPE initial condition. Instead,
the optimizer starts from random global pulse parameters and uses slot-wise
GRAPE gradients chained to `A`, `omega`, `phi0`, and the final phase parameter.
The paper profile is only the reference used to judge whether the optimized
parameters have been recovered.

The default gate-propagation Hamiltonian is now the two-atom 9D two-photon
ladder in the blue-phase gauge, not the earlier 4D effective blockade model.
The phase profile `phi(t)` is applied to the 420 nm `|1> <-> |e>` leg while the
1,013 nm red leg is fixed. The single-atom detuning-gauge Eq. (2) Hamiltonian is
also implemented for dark/bright-state diagnostics; in that gauge the
time-dependent detuning is `delta(t) = -d phi / dt`.

## Dark-State Hamiltonian

The single-atom ladder Hamiltonian from Methods Eq. (2) is implemented in the
`{|1>, |e>, |r>}` basis:

```math
H =
\begin{pmatrix}
0 & \Omega_b/2 & 0 \\
\Omega_b/2 & -\Delta & \Omega_r/2 \\
0 & \Omega_r/2 & -\delta
\end{pmatrix}.
```

Here `Omega_b` is the 420 nm leg and `Omega_r` is the 1013 nm leg. The code also
returns the leading-order dark, bright, and mostly-intermediate eigenvectors
used for scattering diagnostics. The phase convention follows the Methods text:
the time-dependent two-photon detuning is `delta(t) = -d phi / dt`.

## Run Command

```bash
./.venv/bin/python experiments/reproduce_evered2023_parallel_cz_gate.py
./.venv/bin/python scripts/plot_evered2023_parallel_cz.py
```

This writes:

```text
artifacts/evered2023_parallel_cz/evered2023_parameterized_grape_scan.json
artifacts/evered2023_parallel_cz/evered2023_parameterized_grape_summary.png
```

The JSON stores the random-restart parameterized GRAPE time scan. The PNG shows
the fidelity-vs-time curve, the recovered phase profile against Eq. (1), the
recovered pulse parameters, and 9D two-photon ladder population dynamics.

## Experimental Inputs Tracked

The code records the paper-level scale and platform assumptions that are needed
before moving beyond the ideal exact-gate check:

- atom platform: `87Rb`
- Rydberg state: `n = 53`
- two-photon Rabi rate: `Omega / 2pi = 4.6 MHz`
- reported parallel two-qubit CZ fidelity: `99.5%`
- up to `60` atoms operated in parallel
- dark-state sign criterion: suppress intermediate scattering by choosing
  opposite signs for intermediate detuning `Delta` and initial two-photon
  detuning `delta`

Not yet hard-coded because they must be supplied or independently calibrated:
the exact 420/1013 nm Rabi split, intermediate detuning, finite laser rise time,
AOM transfer function, measured laser noise spectra, Doppler/motional traces,
finite-blockade corrections, intermediate-state scattering rate, Rydberg decay,
and SPAM/readout/benchmarking layers.
