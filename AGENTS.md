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
- `notebooks/`: notebook-first development, debugging, validation, and benchmark records.
- `Slides/`: LaTeX slide reports that document completed development before integration.
- `data/`: imported or processed external data used by scripts.
- `artifacts/`: committed JSON/PNG outputs, grouped by version or reproduction line.
- `patches/`: project-owned compatibility patches for external code.
- `rydcalc/`: ThompsonLabPrinceton `rydcalc` Git submodule, pinned as an upstream dependency for future MQDT/Rydberg calculations.

Do not treat `__pycache__`, virtual environments, local build directories, or machine-specific configuration as project logic.

## Development Workflow

All substantive development must follow this three-stage gated workflow. These
stages are review gates, not an instruction to complete all three in one
assistant turn. After finishing each stage, stop and ask the user to review the
artifact before moving to the next stage. Do not start the next stage until the
user explicitly approves it.

1. Develop and debug first in `notebooks/` as a Jupyter notebook. The notebook must include Markdown cells that explain the overall development plan, the goal and role of each code section, and the underlying implementation principle. If the work involves a physical model, the Markdown must describe the model in detail before presenting numerical results, including the Hilbert-space basis or reduced basis, active computational subspace, Hamiltonian terms, controls, pulse envelopes, detunings, dissipation channels, noise parameters, unit conventions, time scales, and fidelity or diagnostic definitions.
   Stop after this stage and wait for user review and approval of the notebook
   and generated artifacts.
2. After notebook testing and development are complete, and only after the user
   approves the notebook stage, write a LaTeX slide report under `Slides/`. The
   slides must summarize the development goal, overall approach, code
   principles, detailed physical model when applicable, numerical assumptions,
   and the executed code results. Include the important plots, tables, metrics,
   and artifact paths needed to understand the outcome without rerunning the
   notebook. Stop after this stage and wait for user review and approval of the
   slide report.
3. Only after the notebook and slide report are complete and the user approves
   both previous stages, package the new functionality into the current
   codebase. Integrate reusable code into the appropriate `src/neutral_yb/`
   module, `experiments/` entry point, `scripts/` helper, and `tests/`
   coverage according to existing project patterns. Notebook code is the
   development record, not the final reusable implementation.

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

Every experiment that produces numerical results must also produce human-readable plots in the same artifact directory. At minimum, include a summary figure showing the key physics outcome, such as fidelity/infidelity versus gate time, threshold markers, relevant population or leakage diagnostics, and pulse amplitude/phase or other optimized controls when applicable. JSON outputs are for detailed machine-readable records, parameter provenance, and reproducibility; they are not a substitute for plots.

Record generated artifact paths in PRs or notes. If a command is expensive, include the exact command and important numerical assumptions in the surrounding docs or PR description.

## Notebook Test Policy

Testing and benchmark content should be saved as Jupyter notebooks in `notebooks/`. Each notebook must use Markdown cells to explain the purpose of the test, the exact comparison being made, the expected interpretation of the outputs, and any assumptions or limitations. For active development work, follow the notebook-first workflow above before moving code into importable modules.

For tests involving physical simulation, the notebook must describe the physical model in detail before presenting numerical results. Include the Hilbert-space basis or reduced basis, active computational subspace, Hamiltonian terms, controls, pulse envelopes, detunings, dissipation channels, noise parameters, unit conventions, time scales, and fidelity or diagnostic definitions. The description should be complete enough that a reader can understand precisely what is being simulated without reverse-engineering the code cells.

## Slides Report Policy

Every completed development task must have a LaTeX slide report in `Slides/`
before the new functionality is integrated into the main package. Create this
slide report only after the user has reviewed and approved the completed
notebook stage. Prefer a focused Beamer-style report that records the
motivation, notebook used for development, implementation logic, physical model
details when relevant, validation commands, generated figures, numerical
results, and artifact paths.

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

Before assuming GitHub is up to date, check `git status --short --branch`. A branch marked `[ahead N]` has local commits that still need `git push origin main`; untracked files marked `??` and dirty submodule contents are not uploaded by pushing the parent repository.

Pull requests should describe the model or workflow changed, list commands run, note generated artifact paths, and call out numerical or physics assumptions. Include the development notebook path and the `Slides/` LaTeX report path for new functionality. For changes touching external integrations, state whether the submodule remained clean or which patch was applied locally.
