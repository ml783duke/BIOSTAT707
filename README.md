# BIOSTAT 707 Checkpoints 1–3

This repository contains reproducible Python/Jupyter analyses of the PhysioNet Challenge 2012 cohorts. Raw data are ignored by Git and are never modified.

## Data

Download `set-a.tar.gz`, `set-b.tar.gz`, `set-c.tar.gz`, and their matching `Outcomes-*.txt` files from PhysioNet and put them in `data/`. Alternatively, extract each archive into `data/set-a/`, `data/set-b/`, and `data/set-c/`. Checkpoints 1 and 2 use set-a only. Checkpoint 3 applies the frozen Checkpoint 2 model to set-b and set-c before loading either outcome file.

## Reproduction

From the repository root, run `pixi install`, followed by:

- `pixi run --locked checkpoint1` for the descriptive data-quality report.
- `pixi run --locked checkpoint2` for the modeling report.
- `pixi run --locked checkpoint3` for the locked external-validation report.

On the development laptop, Checkpoint 1 takes approximately 3 minutes, Checkpoint 2 takes 10–11 minutes, and Checkpoint 3 takes approximately 6 minutes after the Pixi environment is ready. The first run may take longer while Pixi installs the locked dependencies.

The Checkpoint 2 task rebuilds the Checkpoint 1 modeling table, fits all models, and generates `output/checkpoint2.html`. Its saved handoff artifacts for Checkpoint 3 are `output/cp2_final_model.joblib` and `output/cp2_oof_predictions.csv`. Model tables and figures are also written under `output/`.

The Checkpoint 3 task does not refit or select a model using set-b or set-c. It builds feature tables using the same shared cleaning and aggregation functions as set-a, saves locked risk predictions for both sets, and only then reads `Outcomes-b.txt` and `Outcomes-c.txt` to estimate discrimination, calibration, decision-curve net benefit, classification performance at a set-a-only threshold, subgroup performance, and confident mistakes. The primary review file is `output/checkpoint3.html`; supporting tables include `output/cp3_discrimination.csv`, `output/cp3_validation_metrics.csv`, `output/cp3_clinical_utility.csv`, and the two prediction files.

The locked environment uses Python 3.12 and package versions recorded in `pixi.lock`. The notebook is the report entry point and calls the scripts under `scripts/`; all paths are relative to the repository root.
