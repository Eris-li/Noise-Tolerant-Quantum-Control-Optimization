# Ma 2023 Time-Optimal Two-Qubit Gate Reproduction

This board tracks a separate attempt to reproduce the time-optimal two-qubit gate shown in Ma et al., Nature 622, 279-284 (2023), Fig. 3. Keep it separate from the current `v4/v5` open-system `^171Yb` optimization line until the idealized reproduction is stable.

## Source Scope

- Nature target: Ma et al. report metastable `^171Yb` one- and two-qubit gate fidelities of `0.9990(1)` and `0.980(1)` and label Fig. 3 as "Time-optimal two-qubit gates".
- Data source: the Nature article points to Harvard Dataverse at `https://doi.org/10.7910/DVN/TJ6OIF`.
- Method source: the time-optimal pulse family is based on Jandura and Pupillo, Quantum 6, 712 (2022), which identifies global CZ pulses for Rydberg atoms and cites public pulse data at `https://doi.org/10.6084/m9.figshare.19658427`.

## Initial Reproduction Target

Start with the ideal infinite-blockade global-CZ problem already represented by `src/neutral_yb/models/global_cz_4d.py` and `src/neutral_yb/optimization/global_phase_grape.py`. The first numerical target is the dimensionless CZ duration

```text
T * Omega_max = 7.612
```

This target is not yet the full experimental Nature gate. It verifies the reduced Hamiltonian, phase-gate fidelity objective, constant-amplitude phase modulation, and GRAPE workflow before adding finite blockade, decay, erasure/loss accounting, and experimental calibration.

## Run Commands

Quick smoke run:

```bash
./.venv/bin/python experiments/reproduce_ma2023_time_optimal_2q_gate.py --num-tslots 12 --max-iter 2 --num-restarts 1
```

Full first-pass run:

```bash
./.venv/bin/python experiments/reproduce_ma2023_time_optimal_2q_gate.py --duration 7.612 --num-tslots 99 --max-iter 300 --num-restarts 4 --show-progress
```

Outputs are written to `artifacts/ma2023_time_optimal_2q/`.

## Next Steps

1. Reproduce a high-fidelity ideal pulse near `T * Omega_max = 7.612`.
2. Compare the optimized phase shape against the public Jandura-Pupillo pulse data.
3. Convert the dimensionless duration to physical time using the relevant Nature/Yb Rabi calibration.
4. Add finite blockade and Rydberg decay, then compare the simulated two-qubit fidelity with the Nature `0.980(1)` benchmark.
