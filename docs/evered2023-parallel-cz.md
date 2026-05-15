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
- Fidelity and repeated-gate benchmark helpers:
  [evered2023_benchmarking.py](../src/neutral_yb/models/evered2023_benchmarking.py)
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

The reproduction script also supports `--include-paper-seed`, which inserts the
paper Eq. (1) parameters as the first optimizer start before any random
restarts. This is the preferred diagnostic when checking whether the current
Hamiltonian has a local high-fidelity basin connected to the published pulse.

## Fidelity Conventions

The repository now keeps two fidelity conventions explicit:

1. `diagonal_cz_process_fidelity` is the deterministic 4D computational process
   fidelity for the symmetry-reduced diagonal map
   `diag(1, alpha, alpha, beta)`. This is the differentiable objective used by
   the closed-system GRAPE code.
2. `evered2023_exponential_decay_fidelity_from_diagonal_map` applies the
   paper-style repeated-gate convention to a simulated diagonal CZ map: compute
   fidelities for repeated CZ applications and fit them to
   `fidelity(N)=A*F_CZ**N` without an offset.

`diagonal_cz_average_gate_fidelity` remains available as a legacy/reporting
helper, but it is no longer the closed-system GRAPE objective.

### 2026-05-13 fidelity migration note

Earlier revisions used the 4D computational average gate fidelity

```math
F_{\mathrm{avg}} =
\frac{|\mathrm{Tr}(U_{\mathrm{target}}^\dagger M)|^2
      + \mathrm{Tr}(M^\dagger M)}
     {d(d+1)},\quad d=4,
```

implemented for the reduced diagonal map as

```text
(|1 + 2 exp(-i theta) alpha - exp(-2i theta) beta|^2
 + 1 + 2 |alpha|^2 + |beta|^2) / 20.
```

At the user's request, the closed-system GRAPE objective was changed to process
fidelity,

```math
F_{\mathrm{pro}} =
\frac{|\mathrm{Tr}(U_{\mathrm{target}}^\dagger M)|^2}{d^2}
=
\frac{|1 + 2 e^{-i\theta}\alpha - e^{-2i\theta}\beta|^2}{16}.
```

The model method name `phase_gate_fidelity` was intentionally kept for
backwards API compatibility, but it now returns process fidelity in the
closed-system CZ models. The old average-gate metric is available through
`phase_gate_average_fidelity` where implemented. The analytic gradients in
`global_phase_grape`, `amplitude_phase_grape`, `linear_control_grape`, and the
Evered parameterized optimizers were updated at the same time; if a future
compatibility issue appears, check that any new optimizer uses the process
fidelity derivative, not the old population-term derivative.

The second convention matches the extraction style described in Evered et al.
Methods for the experimental `F_CZ`, but it is still a deterministic synthetic
benchmark. A full paper-level reproduction additionally requires the actual
Bell-state/global-randomized circuits, SPAM/loss handling, shot sampling, and
the microscopic error model.

The calibrated Evered reproduction now uses the two-atom 9D two-photon ladder
in the detuning gauge. The Eq. (1) phase profile is represented as
`delta_waveform(t) = -d phi / dt`; the paper-Rabi calibration additionally
uses the leading-order resonance shift

```text
delta_res = (Omega_r^2 - Omega_b^2) / (4 Delta).
```

For the reported `Omega_b/2pi = 237 MHz`, `Omega_r/2pi = 303 MHz`,
`Delta/2pi = 7.8 GHz`, and `Omega/2pi = 4.6 MHz`, this is
`delta_res/Omega = 0.248327759197`. The propagated Hamiltonian is

```text
H(t) = H_drift + [delta_waveform(t) + delta_res] H_delta.
```

The calibrated paper-Rabi run also scores in the leading-order blue-light
dressed computational branches,

```text
|01_tilde> = normalize(|01> + eps |0e>)
|11_tilde> = normalize(|11> + sqrt(2) eps |W_e> + eps^2 |ee>)
eps = Omega_b / (2 Delta).
```

This is the minimal single-intermediate-state implementation of the two-photon
resonance calibration discussed in
[evered2023_9d_from_scratch_validation.ipynb](../notebooks/evered2023_9d_from_scratch_validation.ipynb).
The older blue-phase-gauge 9D path is still available for diagnostics, but it
is not the default interpretation of the calibrated paper-Rabi result.

### Resonance-calibration warning

Do not reuse the paper-reported `237/303 MHz` single-photon Rabi split in a
single-intermediate-state Eq. (2) model with the bare two-photon detuning zero
left at `delta = 0`. In that model, unequal `Omega_b` and `Omega_r` move the
effective two-photon resonance through the differential light shift above. If
the shift is omitted, the paper Eq. (1) pulse is evaluated off resonance and
shows a large coherent error that is an artifact of the model mapping, not a
failure of the published pulse.

For future open-system or more experimental Evered models:

- compute and apply the appropriate two-photon resonance reference before
  judging Eq. (1) fidelity;
- use the simple `delta_res = (Omega_r^2 - Omega_b^2) / (4 Delta)` only for
  the single-intermediate-state leading-order check;
- replace that formula with a diagonalization/light-shift calibration when the
  model includes multiple intermediate states, Clebsch-Gordan factors,
  additional Rydberg levels, or measured AC Stark shifts;
- document whether reported `delta(t)` values are in the bare Hamiltonian
  frame or in the experimentally calibrated two-photon-resonance frame.

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
./.venv/bin/python experiments/reproduce_evered2023_parallel_cz_gate.py \
  --model two-photon-detuning \
  --rabi-calibration paper \
  --light-shift-resonance \
  --dressed-basis \
  --include-paper-seed
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
