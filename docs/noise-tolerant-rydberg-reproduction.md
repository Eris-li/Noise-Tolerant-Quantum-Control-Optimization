# Noise-Tolerant Rydberg Gate Reproduction

This is the early-stage project map for the new noise-tolerant reproduction
line. It covers two user-selected references:

1. Zhang et al., "Logical Qubits with Erasure Conversion Using Metastable
   Neutral Atoms", Nature Physics 22, 910-916 (2026),
   DOI `10.1038/s41567-026-03309-0`.
2. Jandura, Thompson, and Pupillo, "Optimizing Rydberg Gates for
   Logical-Qubit Performance", PRX Quantum 4, 020336 (2023),
   DOI `10.1103/PRXQuantum.4.020336`.

## Working Layout

- `notebooks/noise_tolerant_rydberg_gate_reproduction.ipynb`:
  executable physical-gate baseline for the blockade-limit Hamiltonian,
  local-phase-optimized process fidelity, and starter amplitude/detuning
  scans.
- `Slides/noise_tolerant_rydberg_reproduction/main.tex`:
  Beamer report tying the physical model, erasure-conversion motivation,
  current notebook outputs, and next reproduction milestones together.
- `Slides/noise_tolerant_rydberg_reproduction/main.bib`:
  user-maintained bibliography source. Codex reads keys from this file but
  does not add or rewrite entries.

## Reproduction Stages

1. Physical-gate baseline:
   reproduce the blockaded Rydberg CZ Hamiltonian, diagonal process fidelity,
   and baseline sensitivity to amplitude and detuning errors.
2. Robust-pulse layer:
   add first-order sensitivity propagation and optimize AR, DR, ADR, and CADR
   pulse families from the PRX Quantum paper.
3. Erasure-conversion layer:
   map Rydberg leakage/decay and metastable-state detection outcomes into
   located erasure-like events and unlocated computational errors for the
   Nature Physics result line.
4. Logical-performance layer:
   only after the physical channel is verified, add small-code logical
   performance surrogates and compare physical infidelity against erasure bias.

## Current Baseline Output

The starter notebook currently records these values from the time-optimal seed
baseline:

- `Omega_max*T = 7.634070`
- nominal process fidelity `0.999993035320`
- nominal leakage-like loss `6.959e-06`
- minimum common-amplitude scan fidelity `0.978025007390`
- minimum Fig. 2(b)-style single-atom `Delta_1` scan fidelity
  `0.802681536884`

The notebook also keeps the first Time Optimal/AR/DR attempt as a diagnostic
record:

- Time Optimal: nominal fidelity `0.999993035320`, common `+0.04`
  amplitude-error infidelity `5.221e-03`, Fig. 2(b)-style detuning-probe
  infidelity `3.743e-02`.
- AR first pass: nominal fidelity `0.988494437919`, common `+0.04`
  amplitude-error infidelity `4.979e-02`, detuning-probe infidelity
  `2.263e-02`.
- DR first pass: nominal fidelity `0.990455136578`, midpoint sign-reversal
  detuning-probe infidelity `2.421e-02`.

These AR/DR first-pass values are diagnostic, not successful paper
reproductions. The earlier reduced-basis phase correction and finite-difference
proxy did not reproduce the published robust-pulse advantage.

## Current Robust Optimization Output

The notebook now includes a bounded finite-probe robust optimization pass. This
is no longer a post-hoc sensitivity scan: the optimization objective explicitly
contains the common-amplitude and Fig. 2(b)-style single-atom detuning probe
points.

- Time Optimal at `Omega_max*T = 14.32`: optimizer seed fidelity
  `0.999999761162`.
- AR finite-probe pass at common amplitude `epsilon = +/-0.05`:
  - nominal long-seed worst endpoint fidelity: `0.998262227353`;
  - AR finite-probe worst endpoint fidelity: `0.998552745966`;
  - AR nominal fidelity after robust balancing: `0.999927393819`.
- DR finite-probe pass with midpoint sign reversal of `Delta_1` at
  `Delta_1/Omega_max = +/-0.20`, with `Delta_2=0`:
  - Time Optimal worst endpoint fidelity: `0.979043468996`;
  - DR finite-probe worst endpoint fidelity: `0.980303143863`;
  - DR nominal fidelity after robust balancing: `0.998971262643`;
  - half-probe `|Delta_1|/Omega_max = 0.10` fidelity remains about `0.99481`.

Interpretation: this finite-probe result is a historical diagnostic, not the
current reproduction route. The current notebook uses direct slot-wise phase
optimization instead of a reduced-basis search. The next technical step remains
the paper's half-pulse, sign-reversal DR construction with all phase slots in
the chosen discretization optimized directly.

## Paper-Style First-Order Objective Output

The current notebook now implements the PRX Quantum first-order sensitivity
state construction. The reusable propagation helper is in
`src/neutral_yb/analysis/noise_tolerant_rydberg.py`, with tests checking that
the propagated `psi^(1)` states match central finite differences for both
common amplitude noise and atom-resolved, midpoint sign-reversed detuning.

The corrected paper-structured pass uses the PRX Quantum pulse construction:
AR is optimized as a full CZ pulse, while DR and ADR are optimized as
`controlled-Rz(pi/2)` half-pulses and then applied twice with Doppler detuning
reversed in the second half. The current notebook no longer imports a fixed TO
pulse or any PRX Quantum optimized pulse. It uses a from-zero scan with
deterministic random phase profiles for TO, AR, DR, and ADR. TO is optimized
only against the nominal full-CZ objective and is the sole non-robust baseline.

The notebook-local optimizer now supplies analytic residual Jacobians to
`scipy.optimize.least_squares`. The nominal branch derivatives use the same
prefix/suffix plus `expm_frechet` structure as the packaged GRAPE optimizers.
The AR/DR/ADR first-order sensitivity derivatives use an augmented generator
for the coupled zero-order and first-order states, so no standalone
`noise_tolerant_phase_grape.py` module is introduced at this stage.

The notebook has been changed in place from the earlier checked 32-slot run to
a smoothness-regularized staged setting intended to finish on the order of four
hours. The default stage control is `run_stages = ("ALL",)`, so the notebook
optimizes a lightweight smooth TO baseline and heavier smooth AR/DR/ADR
candidates in one run. The default numerical settings are:

- `num_half_tslots = 96`;
- TO uses one random start, nominal `max_nfev = 300`, and only
  `Omega_max*T = [7.634070]`;
- AR uses four random starts per duration, nominal `max_nfev = 800`, robust
  `max_nfev = 1200`, and
  `Omega_max*T = [14.32, 14.70]`;
- DR uses four random starts per half-duration, nominal/robust
  `max_nfev = 800`, with
  `Omega_max*T_half = [7.70, 8.20]`;
- ADR uses four random starts per half-duration, nominal/robust
  `max_nfev = 800`, with
  `Omega_max*T_half = [10.50, 11.50]`;
- smoothness regularization uses wrapped first- and second-difference phase
  penalties with weights `(0.020, 0.015)` for TO and `(0.060, 0.040)` for
  AR/DR/ADR.

The previous 32-slot diagnostic results are no longer treated as current
outputs. The notebook outputs were cleared after changing the settings so that
the next execution records only production-scale from-zero optimization
results. Stage checkpoints are written under
`notebooks/artifacts/prxq2023_smooth_4h/`, including
`to_ar_phases.npz`, `to_ar_records.json`, and `available_summary.json` after
the run.

The repeated-half DR/ADR construction assumes a real Doppler-reversal
mechanism between the two identical halves. In the switch method the laser
direction is reversed, so `k` changes sign. In the wait method the sequence
waits for `pi / omega_trap`, half of the motional period, so the velocity
reverses and `Delta_j = k v_j` changes sign. The notebook models this as an
ideal sign flip in the detuning trace; it does not simulate the idle wait
dynamics.

The staged production run is designed to reproduce the qualitative Fig. 2
advantages while avoiding visually abrupt phase profiles. AR should flatten the
common-amplitude scan near `|epsilon| <= 0.05` compared with TO, while DR/ADR
test whether repeated half-pulse first-order conditions flatten the Doppler
scan with `Delta_2 = 0`. Because no optimized paper pulse is imported, the
reproduction still depends on the local optimizer finding those basins from
random starts.
