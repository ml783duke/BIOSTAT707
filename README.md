# BIOSTAT 707 Checkpoint 1

This repository contains a reproducible Python/Jupyter analysis of the PhysioNet Challenge 2012 set-a ICU cohort. Raw data should be placed under `data/set-a/` with `data/Outcomes-a.txt`; raw data are ignored by Git and are never modified. From the repository root, install the locked environment with `pixi install`, then execute the complete analysis with `pixi run --locked checkpoint1`. The run writes `output/checkpoint1.html`, `output/set-a_long.csv`, `output/set-a_wide.csv`, summary tables, and missingness figures.
