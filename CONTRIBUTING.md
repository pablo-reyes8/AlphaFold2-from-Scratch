# Contributing

Thank you for contributing to this AlphaFold2-from-scratch repository.

This project is a research-oriented implementation focused on clarity, modularity, and reproducibility. Contributions are welcome across the full pipeline: data ingestion, model architecture, geometry-aware losses, training utilities, distributed training, notebooks, documentation, and tests.

## What We Appreciate

Useful contributions usually fall into one of these categories:

- Bug fixes in the data pipeline, model modules, losses, or training stack
- Additional tests for tensor shapes, numerical stability, or structural invariants
- Documentation improvements for the architecture, CLI workflows, or notebook usage
- Better reproducibility in configs, scripts, and small showcase datasets
- Performance or ergonomics improvements that preserve the current design goals

## Before You Start

Please keep a few project-specific constraints in mind:

- Prefer small, focused pull requests over large mixed refactors.
- Do not commit large datasets, checkpoints, or generated artifacts.
- Keep the bundled showcase data small and reproducible.
- Preserve the modular structure of the architecture instead of hiding logic in notebooks.
- Avoid changing public training or data interfaces unless the benefit is clear and documented.

## Development Setup

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional editable install for local CLI entry points
pip install -e '.[dev,data]'
```

If you work from Conda, use the equivalent environment setup and install the same requirements.

## Contribution Workflow

1. Fork the repository and create a focused branch.
2. Make the smallest coherent change that solves the problem.
3. Add or update tests when behavior changes.
4. Update documentation when CLI behavior, configs, or user-facing workflows change.
5. Open a pull request with a clear summary, validation notes, and any tradeoffs.

## Validation Expectations

Run the narrowest relevant checks first, then broader checks before opening the pull request.

General regression check:

```bash
python3 -m pytest --capture=no tests -q
```

Model wiring and synthetic validation:

```bash
python3 scripts/validate_model.py instantiate --config config/experiments/af2_poc.yaml --device cpu
python3 scripts/validate_model.py forward-smoke --config config/experiments/af2_poc.yaml --device cpu
```

Data pipeline sanity check with the bundled showcase subset:

```bash
python3 scripts/prepare_data.py loader-smoke \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --batch-size 2 \
  --max-samples 2
```

Use targeted test files when a change is local, but do not skip the broader suite for risky architecture or training edits.

## Coding Guidelines

- Keep module boundaries explicit and easy to inspect.
- Prefer readable, well-scoped functions over hidden side effects.
- Add module docstrings and concise comments when they improve clarity.
- Keep imports clean and avoid introducing unused dependencies.
- Preserve CPU-friendly smoke-test paths when adding GPU-oriented features.
- Follow the existing config-driven approach instead of hardcoding experiment settings.

## Data Changes

For data-related contributions:

- Keep paths manifest-driven whenever possible.
- Prefer adding CLI-visible validation instead of notebook-only steps.
- Do not check in large benchmark snapshots.
- If a tiny real-data sample is needed, keep it small enough to remain practical for version control.

## Model and Training Changes

For architecture or optimization changes:

- Avoid breaking the current Evoformer, structure, and loss contracts unless the change is deliberate and documented.
- Add tests for tensor shapes, finite outputs, masking behavior, and deterministic evaluation where applicable.
- If you touch distributed training, include the intended launch mode and expected hardware assumptions in the pull request description.

## Pull Request Quality Bar

A good pull request for this repository usually includes:

- A clear problem statement
- A concise description of the solution
- Validation commands that were actually run
- Notes on any unresolved risks or follow-up work
- Documentation updates when the user workflow changed

If your change is exploratory or incomplete, mark that clearly in the pull request so reviewers know how to interpret it.

## Reporting Bugs

Please use the issue templates when opening a bug report or feature request. Structured reports make it much easier to reproduce problems in data ingestion, geometry modules, losses, and distributed training.

For security-sensitive issues, do not open a public issue. Follow the private reporting guidance in [SECURITY.md](SECURITY.md).
