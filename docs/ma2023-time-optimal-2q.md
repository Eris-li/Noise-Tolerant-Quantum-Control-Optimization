# Ma 2023 Time-Optimal Two-Qubit Gate Reproduction

This board tracks a separate attempt to reproduce the time-optimal two-qubit gate shown in Ma et al., Nature 622, 279-284 (2023), Fig. 3. Keep it separate from the current `v4/v5` open-system `^171Yb` optimization line because Ma 2023 uses a metastable-qubit Rydberg gate, not the later clock-shelving model.

## Source Scope

- Nature target: Ma et al. report metastable `^171Yb` one- and two-qubit gate fidelities of `0.9990(1)` and `0.980(1)` and label Fig. 3 as "Time-optimal two-qubit gates".
- Data source: the Nature article points to Harvard Dataverse at `https://doi.org/10.7910/DVN/TJ6OIF`.
- Method source: the time-optimal pulse family is based on Jandura and Pupillo, Quantum 6, 712 (2022), which identifies global CZ pulses for Rydberg atoms and cites public pulse data at `https://doi.org/10.6084/m9.figshare.19658427`.
- Extended Data Fig. 1 states that the detuning between addressed and neighboring Rydberg states is `5.8 * Omega_UV`; this is tracked in the calibration layer for later off-resonant leakage modeling.

## Implemented Model

The current Ma-specific implementation is:

- Methods-level physical model: `src/neutral_yb/models/ma2023_six_level.py`
- legacy reduced model: `src/neutral_yb/models/ma2023_time_optimal_2q.py`
- calibration/ensemble builder: `src/neutral_yb/config/ma2023_calibration.py`
- pulse family and validation: `src/neutral_yb/models/ma2023_pulse.py`
- six-level phase-only optimizer: `src/neutral_yb/optimization/ma2023_six_level_grape.py`
- data importer: `scripts/import_ma2023_dataverse.py`
- fixed-pulse evaluator: `experiments/evaluate_ma2023_fig3_pulse.py`
- method-first reproducer: `experiments/reproduce_ma2023_from_method.py`
- Methods-level six-level reproducer: `experiments/reproduce_ma2023_six_level_from_method.py`
- scan entry point: `experiments/scan_ma2023_time_optimal_2q_open_system.py`
- plotting entry point: `scripts/plot_ma2023_time_optimal_2q.py`
- method/data comparator: `scripts/compare_ma2023_method_to_dataverse.py`

The Methods-level model uses the single-atom basis `|0>`, `|1>`, `|r_-3/2>`, `|r_-1/2>`, `|r_1/2>`, `|r_3/2>`. In the perfect-blockade limit it removes double-Rydberg states and evolves the `|00>`, `|01>`/`|10>`, and `|11>` initial states in three five-dimensional subspaces, matching the decomposition described in the Methods text. The previous seven-state model is kept as a legacy reduced comparison.

The local Dataverse CSV files are normalized into `data/ma2023/processed/fig3_time_optimal_gate.json`. The importer derives the Fig. 3 pulse duration `1.239967 us`, peak UV Rabi rate `1.59 MHz`, and dimensionless duration `T * Omega_max = 12.3876` using angular-frequency units.

The method-first reproducer now uses the same high-level control family described in the article Methods for two-qubit entangling gates: total duration `T` is fixed, `Omega(t)` is a fixed Gaussian-edge envelope, and the optimizer varies the phase-rate Chebyshev coefficients plus the global and single-qubit phase parameters. The default parameterization is `dot(phi)(t) = sum_n c_n T_n(2t/T - 1)` with `n <= 13`, sampled on `N = 100` slots and integrated to the piecewise-constant phases used by the propagator.

For Fig. 3-style trajectory plots, the code reports Rydberg-transition Bloch vectors in the `|01>-|0r>` and `|11>-|W_r>` subspaces. The older computational active-subspace projection is still saved as `computational_active_bloch_*`, but it is not a useful `x-y` trajectory for `|01>` or `|11>` inputs because those states do not develop coherence between `|01>` and `|11>` during a diagonal phase gate.

## Hamiltonian And Noise Audit

The current Methods-level coherent Hamiltonian is

```text
H = H_d + u_x(t) H_x + u_y(t) H_y
u_x = Omega(t) cos(phi(t)), u_y = Omega(t) sin(phi(t))
```

with the four Rydberg Zeeman sublevels and Clebsch-Gordon factors from the Methods equation: `r_-3/2` and `r_3/2` couple with coefficient `1/2`, while `r_-1/2` and `r_1/2` couple with coefficient `1/(2 sqrt(3))`. The Rydberg detunings are `-3 Delta_r`, `-2 Delta_r`, `-Delta_r`, and `0`, with `Delta_m = 0`. For Ma 2023 calibration, `Delta_r / Omega = 5.8`.

The perfect-blockade six-level Hamiltonian now matches the Methods-level coherent model, excluding finite-blockade double-Rydberg corrections. Implemented six-level noise terms are Rydberg decay into a loss sink, optional Rydberg dephasing, common detuning, Rydberg Zeeman offset, and Rabi amplitude scale noise. The default Dataverse-derived calibration sets the Fig. 3 Rabi/time scale, `Delta_r / Omega = 5.8`, and the measured Rydberg lifetime `T1,r = 65 us`.

Missing or approximate relative to the full noise model: finite but imperfect van der Waals blockade, Doppler/motional Monte Carlo detuning traces, measured laser phase and intensity noise spectra, tweezer/light-shift noise, measured AOM transfer-function and closed-loop phase correction, atom-position-dependent Rabi and blockade variation, blackbody/branching-resolved Rydberg decay channels, and benchmark/readout/SPAM layers. These should be added before claiming full Methods-level noise-model agreement.

## Noise Model Implementation

The six-level model separates noise into three categories:

- Markovian Rydberg decay: implemented as Lindblad collapse operators from every Rydberg basis state into two sink states, `detected_decay` and `undetected_decay`. These are not optical dark/bright dressed states; they are bookkeeping states for decay products that are detected or not detected by the erasure readout. The total rate is `1 / (Omega_ref * T1,r)`; the default `T1,r = 65 us` comes from Ma 2023. The default detected fraction is `0.5`, matching the Methods statement that only about half of Rydberg decays are detected through `1S0`.
- Coherent static errors: implemented as Hamiltonian parameters or per-trajectory offsets, including common detuning, Rydberg Zeeman offset, and Rabi amplitude scaling.
- Non-Markovian errors: represented by Monte Carlo traces in `src/neutral_yb/models/ma2023_noise.py`. A trace contains slot-wise `common_detuning`, `rabi_scale`, and `phase_offset`. The six-level optimizer can evolve density matrices with these traces using exact slot-wise Lindblad propagators.

For Doppler/quasistatic detuning, the helper `doppler_detuning_rms_from_t2_star()` converts a Gaussian Ramsey `T2*` into a dimensionless RMS detuning using `sigma_delta = sqrt(2) / T2*`. Ma 2023 reports `T2* = 5.7 us`; the resulting dimensionless RMS should be used as the Monte Carlo `quasistatic_detuning_rms`.

Laser phase noise should be simulated as a sampled phase offset added to `phi(t)` before forming `u_x = Omega cos(phi)` and `u_y = Omega sin(phi)`. Laser intensity noise should be simulated as a multiplicative Rabi scale applied to the pulse amplitude. The current code supports white or quasistatic trace injection; measured spectra can be converted to colored time traces later without changing the simulator interface.

## Run Commands

Import/normalize the local Dataverse CSV files:

```bash
./.venv/bin/python scripts/import_ma2023_dataverse.py
```

Quick smoke run:

```bash
./.venv/bin/python experiments/evaluate_ma2023_fig3_pulse.py --num-tslots 32 --ensemble-size 1 --output ma2023_fig3_pulse_smoke.json
```

Fixed Fig. 3 pulse evaluation:

```bash
./.venv/bin/python experiments/evaluate_ma2023_fig3_pulse.py --num-tslots 96 --ensemble-size 1 --output ma2023_fig3_pulse_96slot.json
```

Do not use this as the main reproduction. It verifies that the imported pulse and current model propagate correctly.

Methods-level six-level reproduction from a smooth generic initial pulse:

```bash
./.venv/bin/python experiments/reproduce_ma2023_six_level_from_method.py --num-tslots 100 --max-iter 160 --num-restarts 4 --phase-parameterization chebyshev --chebyshev-degree 13 --show-progress
```

Legacy reduced-model method-first reproduction:

```bash
./.venv/bin/python experiments/reproduce_ma2023_from_method.py --num-tslots 96 --max-iter 160 --num-restarts 4 --ensemble-size 1 --show-progress
./.venv/bin/python scripts/compare_ma2023_method_to_dataverse.py
```

High-resolution method-first run:

```bash
./.venv/bin/python experiments/reproduce_ma2023_from_method.py --num-tslots 256 --max-iter 200 --num-restarts 4 --ensemble-size 5 --show-progress
./.venv/bin/python scripts/compare_ma2023_method_to_dataverse.py --summary artifacts/ma2023_time_optimal_2q/from_method/ma2023_from_method_summary.json
```

Method-first outputs are written to `artifacts/ma2023_time_optimal_2q/from_method/`.

## Current Consistency Check

With the imported Fig. 3 pulse and current reduced model:

- `32` slots: active-channel fidelity `0.982887770789468`
- `96` slots: active-channel fidelity `0.9886474886239447`
- paper target: two-qubit gate fidelity `0.980(1)`

This is a successful program/data integration test, but it is not the main reproduction because it uses the published pulse. The method-first reproducer starts from generic smooth controls and uses Dataverse data only for calibration and comparison.

With the phase-only Gaussian-edge method-first reproducer:

- `96` slots, `4` restarts, `220` max iterations: objective fidelity `0.994684718734313`
- active-channel fidelity `0.9929591146914648`
- `max(Omega / Omega_max) = 1.0`, with first and last amplitudes exactly `0.0`
- optimized phase range: `[-1.0457326745429372, 1.0477240883160732]`
- outputs: `artifacts/ma2023_time_optimal_2q/from_method/ma2023_from_method_TOmega_12p388.json` and `ma2023_from_method_comparison.png`

With the Methods-level six-level model:

- direct `N = 100` phase GRAPE reaches fidelity `0.9999999137005409`, or infidelity `8.63e-8`, in `ma2023_six_level_direct_N100_seed31.json`.
- Chebyshev phase-rate `n <= 13` from random initialization reaches only `0.9668751671467305`; this is not sufficient.
- Projecting the high-fidelity direct pulse into Chebyshev-13 and refining reaches `0.9996972210978747`, or infidelity `3.03e-4`, in `ma2023_six_level_cheb13_N100_refined_ideal.json`.
- The Chebyshev-13 result is now implemented and valid, but it has not yet reproduced the Methods claim that truncation at `n_max = 13` has negligible fidelity cost.

The remaining differences from the full paper-level implementation are explicit. The exact experimental Gaussian edge width and closed-loop AOM pre-distortion are not encoded unless supplied as parameters. The main coherent model now includes the four Rydberg Zeeman sublevels in the perfect-blockade approximation, but finite-blockade double-Rydberg dynamics and the full Monte Carlo noise traces are not yet included. Randomized-circuit-benchmarking extraction and SPAM/readout corrections are not yet simulated.

## Next Steps

1. Run the six-level optimizer at `N = 100` and tune the phase regularization until the Methods-level ideal infidelity approaches the reported `< 1e-5` target.
2. Add Doppler, laser phase-noise, and intensity-noise Monte Carlo traces from independently measured parameters.
3. Add finite-blockade double-Rydberg corrections if needed beyond the perfect-blockade Methods optimization.
4. Add the randomized-circuit-benchmarking extraction layer later, after the physical gate model is stable.
