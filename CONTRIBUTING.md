# Contributing

This repository is a research-code project for neutral-atom quantum-control
models, reference reproductions, and numerical experiments. Contributions are
welcome when they keep the physical assumptions, numerical settings, and
validation path explicit.

## Development Setup

Use Python 3.12.

```bash
git submodule update --init --recursive
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .
```

Run the default test suite with:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Optional `rydcalc` integration requires the submodule and local extension build:

```bash
./.venv/bin/python -m pip install -e '.[rydcalc]'
./.venv/bin/python scripts/build_rydcalc_extension.py
```

## Repository Conventions

- Put reusable Python code under `src/neutral_yb/`.
- Put reproducible experiment entry points under `experiments/`.
- Put plotting, import, and environment helper commands under `scripts/`.
- Put focused `unittest` coverage under `tests/`.
- Put explanatory model notes and literature links under `docs/`.
- Keep generated results in versioned, descriptive paths under `artifacts/`.

Avoid changing vendored or submodule code directly. If `rydcalc` or ARC needs a
local compatibility patch, add a documented patch under `patches/` or an adapter
under `src/neutral_yb/external/`.

## Research Contribution Checklist

Before opening a pull request, make sure the change answers these questions:

- What physical model or numerical workflow changed?
- Which basis states, detunings, couplings, noise terms, and time units are used?
- Which prior result, analytic limit, or internal consistency check validates it?
- Which command, notebook, JSON, or figure reproduces the reported result?
- Which tests cover the new behavior?

For exploratory work, prefer a notebook-first record under `notebooks/` before
moving reusable logic into the package.

## Pull Request Checklist

- Keep the change focused.
- Do not overwrite committed artifacts without a clear reason.
- Do not commit virtual environments, caches, local logs, or machine-specific
  paths.
- Run `python -m unittest discover -s tests -v`.
- Update `README.md`, `docs/project-map.md`, or a model note when the public
  workflow changes.
