# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `mmood3d/`, with dataset adapters in `mmood3d/datasets/`, runtime hooks in `mmood3d/engine/`, model code in `mmood3d/models/`, and shared helpers in `mmood3d/utils/`. Training and evaluation entry points are under `tools/` (`train.py`, `test.py`, `create_data.py`). Small verification and data utility scripts live in `scripts/`. Use `data/` for local datasets, `checkpoints/` for model weights, and `results/` for experiment outputs. `notes/` and `second-page/` contain research artifacts and should stay separate from production code changes.

## Build, Test, and Development Commands
Set up the editable package with `pip install -v -e .` inside the project environment. Prepare nuScenes metadata with `python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes`. Train locally with `python tools/train.py mmood3d/configs/ood/ood.py`, or use distributed training with `./tools/dist_train.sh mmood3d/configs/ood/ood.py 8`. Run evaluation with `python tools/test.py <config> <checkpoint>`. For focused debugging, use the script-based checks in `scripts/`, for example `python scripts/test_backbone.py` or `python scripts/test_projection_pipeline.py`.

## Coding Style & Naming Conventions
Follow Python 3.8 style with 4-space indentation and PEP 8-oriented formatting. `setup.cfg` defines `yapf`, `isort`, and `flake8`; keep imports grouped consistently and keep lines near the existing 79-character convention. Use `snake_case` for modules, functions, config files, and script names. Match existing OpenMMLab-style registry patterns instead of introducing new abstractions unless required.

## Testing Guidelines
This snapshot does not contain a dedicated `tests/` package; validation is primarily script- and pipeline-based. Add focused checks near similar files in `scripts/` and name them `test_<feature>.py`. Before opening a PR, run the smallest relevant script test, plus at least one end-to-end command path affected by your change, such as training startup, dataset creation, or inference.

## Commit & Pull Request Guidelines
Usable git history is not present in this workspace snapshot, so follow a simple imperative style for commits, such as `Fix nuscenes OOD label mapping`. Keep each commit scoped to one logical change. PRs should include a short problem statement, the exact commands run for verification, any dataset or checkpoint assumptions, and screenshots or metric snippets when outputs change.

## Data & Configuration Notes
Do not commit datasets, generated annotations, checkpoints, or large result artifacts. Keep machine-specific paths out of tracked configs, and prefer parameterized paths relative to `data/`, `checkpoints/`, and `results/`.
