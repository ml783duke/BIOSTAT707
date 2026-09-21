# BIOSTAT 707 Checkpoint 1

This repository contains a reproducible Python/Jupyter analysis of the [PhysioNet Challenge 2012 set-a cohort](https://physionet.org/content/challenge-2012/1.0.0/). Download `set-a.tar.gz` and `Outcomes-a.txt` from PhysioNet and put both in `data/`. Alternatively, extract the archive so the 4,000 patient files are in `data/set-a/`. The analysis reads the archive when present, otherwise the extracted directory. Raw data are ignored by Git and are never modified.

From the repository root, run `pixi install`, then `pixi run --locked checkpoint1`. The task executes `checkpoint1.ipynb` and generates `output/checkpoint1.html`, `output/set-a_long.csv`, `output/set-a_wide.csv`, summary CSVs, and figures under `output/figures/`. The notebook is the sole execution entry point and calls `scripts/run_analysis.py` for the data processing. Use relative project paths; run the task from the repository root.

The locked environment uses Python 3.12 and package versions recorded in `pixi.lock`. Open `output/checkpoint1.html` to review the rendered report. Before submitting, review the code and results yourself, update the AI-use statement to reflect your actual checks, and commit only the intended files; raw data and most generated outputs remain untracked by design.
