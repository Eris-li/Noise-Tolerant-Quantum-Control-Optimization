# Repository Guidelines

## Project Scope

This is a Python 3.12 research package for neutral `^171Yb` quantum-control simulations. The active work is multi-qubit gate modeling and optimization, especially `CZ` gates, with later extensions toward `CNOT` and three-qubit gates. Single-qubit gates are not a project target unless they are part of a composite multi-qubit workflow.

The repository keeps several version lines alive for comparison:

- `v1`: frozen ideal 4D global `CZ` reference.
- `v2`: closed-system finite-blockade 5D correction model.
- `v3`: closed-system two-photon 9D model with explicit intermediate state.
- `v4`: open-system `^171Yb clock -> Rydberg` full-gate model with fixed shelving/unshelving and optimized UV segment.
- `v5`: current calibrated `^171Yb` scan line with strict-literature and experimental-surrogate profiles.
- `ma2023_time_optimal_2q`: independent Ma et al. Nature 2023 metastable-qubit Rydberg-gate reproduction line.

## Project Structure

Core importable code lives in `src/neutral_yb/`.

- `config/`: species definitions, artifact paths, `^171Yb` calibration profiles, and Ma 2023 calibration helpers.
- `models/`: Hamiltonian/open-system models, including historical references and active `^171Yb`/Ma 2023 models.
- `optimization/`: GRAPE and open-system optimization routines.

Operational code and outputs are organized separately:

- `tests/`: deterministic `unittest` coverage.
- `experiments/`: scan, optimization, validation, and benchmark entry points. These may be long-running.
- `scripts/`: plotting, data import, and local environment helper scripts.
- `docs/`: project maps, model notes, version history, references, and integration notes.
- `data/`: imported or processed external data used by scripts.
- `artifacts/`: committed JSON/PNG outputs, grouped by version or reproduction line.
- `patches/`: project-owned compatibility patches for external code.
- `rydcalc/`: ThompsonLabPrinceton `rydcalc` Git submodule, pinned as an upstream dependency for future MQDT/Rydberg calculations.

Do not treat `__pycache__`, virtual environments, local build directories, or machine-specific configuration as project logic.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: fetch the `rydcalc` submodule after cloning.
- `python3 -m venv .venv`: create a Linux/WSL virtual environment.
- `./.venv/bin/python -m pip install -r requirements.txt`: install runtime dependencies.
- `./.venv/bin/python -m pip install -e .`: install the package in editable mode.
- `./.venv/bin/python -m pip install -e '.[rydcalc]'`: install optional dependencies needed for local `rydcalc` validation.
- `./.venv/bin/python -m unittest discover -s tests -v`: run the full local test suite.
- `docker compose build`: build the containerized development/test image.
- `docker compose run --rm test`: run tests inside Docker.
- `docker compose run --rm smoke-v4`: run the `v4` open-system smoke workflow.

On Windows PowerShell, use `scripts/create_venv.ps1` and `scripts/run_python.ps1`.

## Coding Style

Use 4-space indentation, type hints, and `from __future__ import annotations` for new Python modules. Prefer dataclasses for immutable model/config containers and explicit NumPy dtypes for complex arrays. Module and script names should be lowercase snake_case, e.g. `two_stage_scan_yb171_v5_0_300ns_10mhz.py`. Test files should follow `tests/test_*.py`.

No formatter is configured. Keep imports grouped as standard library, third-party, then local package. Follow existing model and optimizer patterns before introducing new abstractions.

## Testing Guidelines

Add focused `unittest.TestCase` coverage for new models, calibration helpers, objective functions, and optimizer behavior. Keep default discovery deterministic and reasonably fast; put expensive scans in `experiments/` or Docker smoke workflows.

If tests need package import bootstrapping, follow:

```python
from tests import _bootstrap  # noqa: F401
```

When changing open-system dynamics or gradients, prefer at least one small shape/fidelity test and one finite-difference or consistency check when feasible.

## Experiment and Artifact Policy

Do not overwrite committed artifacts casually. Write new scan outputs to descriptive, versioned subdirectories under `artifacts/`, usually through `neutral_yb.config.artifact_paths`.

Record generated artifact paths in PRs or notes. If a command is expensive, include the exact command and important numerical assumptions in the surrounding docs or PR description.

## Submodule Policy

`rydcalc/` is managed as a Git submodule pinned to the upstream ThompsonLabPrinceton repository. Do not casually edit files inside `rydcalc/`. Prefer adding adapters under `src/neutral_yb/` or documenting local compatibility patches under `patches/`.

For Python 3.12 local validation, prefer building the ARC C extension for the active interpreter and importing `rydcalc` through the project adapter:

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
./.venv/bin/python scripts/build_rydcalc_extension.py
./.venv/bin/python -c "from neutral_yb.external.rydcalc_adapter import build_yb171_atom; print(build_yb171_atom(use_db=False).name)"
```

The adapter in `neutral_yb.external.rydcalc_adapter` provides the NumPy 2 compatibility shim without editing the submodule. Use `patches/rydcalc-python312-numpy2.patch` only as a temporary fallback when explicitly testing a no-extension path, and reverse it afterward:

```bash
git -C rydcalc apply ../patches/rydcalc-python312-numpy2.patch
git -C rydcalc apply -R ../patches/rydcalc-python312-numpy2.patch
```

Use `git submodule status --recursive` to check the pinned upstream commit before reporting `rydcalc` results.

## Commit and Pull Request Guidelines

Recent history uses short, direct commit subjects, often in Chinese, describing the simulation or artifact change. Keep commits focused and mention the affected line when useful, such as `v4`, `v5`, `ma2023`, or `rydcalc`.

Pull requests should describe the model or workflow changed, list commands run, note generated artifact paths, and call out numerical or physics assumptions. For changes touching external integrations, state whether the submodule remained clean or which patch was applied locally.
