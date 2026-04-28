# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 package for neutral `^171Yb` quantum-control simulations. Core importable code lives in `src/neutral_yb/`, with physics models under `models/`, calibration helpers under `config/`, and optimization routines under `optimization/` when present. Tests are in `tests/` and use `unittest`. Long-running scans and validation entry points live in `experiments/`; plotting and environment helpers live in `scripts/`. Committed JSON/PNG outputs are versioned under `artifacts/`. Project notes and model descriptions are in `docs/`.

## Build, Test, and Development Commands

- `python3 -m venv .venv`: create a Linux/WSL virtual environment.
- `./.venv/bin/python -m pip install -r requirements.txt`: install dependencies.
- `./.venv/bin/python -m pip install -e .`: install editable package.
- `./.venv/bin/python -m unittest discover -s tests -v`: run the full local test suite.
- `docker compose build`: build the containerized development/test image.
- `docker compose run --rm test`: run tests inside Docker.
- `docker compose run --rm smoke-v4`: run the `v4` open-system smoke workflow.

On Windows PowerShell, use `scripts/create_venv.ps1` and `scripts/run_python.ps1`.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints, and `from __future__ import annotations` for new Python modules. Prefer dataclasses for immutable model/config containers and explicit NumPy dtypes for complex arrays. Module and script names should be lowercase snake_case, e.g. `two_stage_scan_yb171_v5_0_300ns_10mhz.py`. Test files should follow `tests/test_*.py`. No formatter is configured; keep imports grouped as standard library, third-party, then local package.

## Testing Guidelines

Add focused `unittest.TestCase` coverage for new models, calibration helpers, and optimization objectives. Keep tests deterministic and small enough for default discovery; put expensive scans in `experiments/` or Docker smoke commands. If tests need package import bootstrapping, follow `from tests import _bootstrap  # noqa: F401`.

## Commit & Pull Request Guidelines

Recent history uses short, direct commit subjects, often in Chinese, describing the simulation or artifact change. Keep commits focused and mention the affected version line when useful, such as `v4` or `v5`. Pull requests should describe the model or workflow changed, list commands run, note generated artifact paths, and call out numerical or physics assumptions.

## Artifact & Configuration Notes

Do not overwrite committed artifacts casually. Write new scan outputs to versioned subdirectories under `artifacts/` with descriptive run names. Avoid committing virtual environments, caches, or machine-specific configuration.
