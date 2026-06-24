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
reproductions. The compact Fourier correction and finite-difference proxy did
not reproduce the published robust-pulse advantage.

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

Interpretation: the amplitude-robust compact search gives a small but clear
robustness improvement while preserving high nominal fidelity. The detuning
result is only a modest improvement at the large `0.20` probe, so the next
technical step remains the paper's half-pulse, sign-reversal DR construction
with full GRAPE or the documented pulse basis.

## Paper-Style First-Order Objective Output

The current notebook now implements the PRX Quantum first-order sensitivity
state construction. The reusable propagation helper is in
`src/neutral_yb/analysis/noise_tolerant_rydberg.py`, with tests checking that
the propagated `psi^(1)` states match central finite differences for both
common amplitude noise and atom-resolved, midpoint sign-reversed detuning.

The compact first-order pass uses `Omega_max*T = 14.32`, 40 time slots, and 10
Fourier correction coefficients:

- Time Optimal: `F0 = 0.999999981780`.
- AR first-order robust pass:
  - sensitivity norm drops from `9.636384e+01` to `5.740975e+00`;
  - diagnostic `epsilon = +/-0.05` worst endpoint fidelity improves from
    `0.995662275603` to `0.996558440020`;
  - nominal fidelity remains `0.999991477077`.
- DR first-order robust pass with midpoint sign reversal:
  - diagnostic `Delta_1/Omega_max = +/-0.20`, `Delta_2=0` worst endpoint
    fidelity is `0.906992297543`, worse than the Time Optimal diagnostic
    `0.976711957540`;
  - nominal fidelity drops to `0.986372734744`;
  - projected detuning sensitivity norm changes from `1.917865e+00` to
    `7.005785e+00`.

This diagnostic explains the Fig. 2 discrepancy. The previous notebook compared
against the wrong detuning scan (`Delta_1=Delta_2`) and used an over-constrained
DR first-order target. Those issues are now corrected. The current compact
full-pulse optimizer still does not reproduce the paper's DR curve because the
paper's DR pulse is built as a repeated half-pulse: one CRz(pi/2) pulse is
optimized, then the same pulse is applied again with the Doppler shift reversed.
Implementing that half-pulse construction, with full slot-wise or documented-
basis GRAPE and analytic gradients, is required before claiming
publication-level Fig. 2(b), ADR, or CADR reproduction.
