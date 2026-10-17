# BIOSTAT 707 Checkpoints 1 and 2

This repository contains reproducible Python/Jupyter analyses of the PhysioNet Challenge 2012 set-a cohort. Raw data are ignored by Git and are never modified.

## Data

Download `set-a.tar.gz` and `Outcomes-a.txt` from PhysioNet and put both in `data/`. Alternatively, extract the archive so the 4,000 patient files are in `data/set-a/`. Checkpoint 2 uses set-a only and does not read set-b or set-c.

## Reproduction

From the repository root, run `pixi install`, followed by:

- `pixi run --locked checkpoint1` for the descriptive data-quality report.
- `pixi run --locked checkpoint2` for the modeling report.

On the development laptop, Checkpoint 1 takes approximately 3 minutes and Checkpoint 2 takes 10 minutes after the Pixi environment is ready. The first run may take longer while Pixi installs the locked dependencies.

The Checkpoint 2 task rebuilds the Checkpoint 1 modeling table, fits all models, and generates `output/checkpoint2.html`. Its saved handoff artifacts for Checkpoint 3 are `output/cp2_final_model.joblib` and `output/cp2_oof_predictions.csv`. Model tables and figures are also written under `output/`.

The locked environment uses Python 3.12 and package versions recorded in `pixi.lock`. The notebook is the report entry point and calls the scripts under `scripts/`; all paths are relative to the repository root.
