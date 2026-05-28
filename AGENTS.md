# Repository Guidelines

This repository is a Python 3.12 research package for neutral `^{171}Yb` quantum-control simulations.

## Current Scope

The cleaned repository keeps three reproducible result lines:

1. `v1` noiseless random-initialized GRAPE baseline.
2. `evered2023_parallel_cz` parameterized/basis-function GRAPE reproduction.
3. `^{171}Yb` UV rising/falling-edge scan for the shelved control-Rydberg segment.

Notebooks are historical development records and should not be edited unless the user explicitly asks.

## Directory Roles

- `src/neutral_yb/`: reusable models, optimizers, analysis workflows, and config helpers.
- `experiments/`: thin one-off recipes that call `src`, write artifacts, and record command-line parameters.
- `scripts/`: retained plotting/environment helpers for the kept result lines.
- `artifacts/`: retained JSON/CSV/PNG outputs for the three current result lines.
- `docs/`: project map, version history, references, and focused result-line notes.
- `rydcalc/`: upstream submodule; do not edit it casually.

Do not put reusable logic in `experiments/`. If a workflow will be reused, move the logic into `src/neutral_yb/` and keep the experiment as a small CLI wrapper.

## Useful Commands

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m unittest discover -s tests -v
```

Focused current-line tests:

```bash
./.venv/bin/python -m unittest tests.test_global_cz_4d tests.test_evered2023_parallel_cz tests.test_shelved_cr_phase_grape -v
```

UV edge smoke:

```bash
./.venv/bin/python experiments/scan_yb171_uv_edge_effect.py --smoke
```

## Artifact Policy

Keep only the current-line artifact directories:

- `artifacts/v1/`
- `artifacts/evered2023_parallel_cz/`
- `artifacts/v5/closed_cr_edge_time_optimal_scan/`

Do not overwrite committed artifacts casually. If verification requires regenerating a plot, check whether the binary changed only because of plotting metadata before leaving it modified.

## Coding Style

Use 4-space indentation, type hints, and `from __future__ import annotations` for new Python modules. Prefer dataclasses for immutable model/config containers and explicit NumPy dtypes for complex arrays. Tests should use `unittest`.

## Submodule Policy

`rydcalc/` is a Git submodule. Prefer adapters under `src/neutral_yb/external/` or documented patches under `patches/` instead of direct edits inside the submodule.
