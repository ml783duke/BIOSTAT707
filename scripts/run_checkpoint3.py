"""Perform locked external validation for BIOSTAT 707 Checkpoint 3."""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from run_checkpoint2 import FoldPreprocessor
from modeling_table import build_modeling_table, clean_measurements

# The Checkpoint 2 artifact was serialized when its module ran as __main__.
# Register the class under that name so the frozen artifact remains loadable.
setattr(sys.modules["__main__"], "FoldPreprocessor", FoldPreprocessor)

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = OUT / "figures"
TARGET = "In-hospital_death"
OUTCOME_COLUMNS = {"SAPS-I", "SOFA", "Length_of_stay", "Survival", TARGET}
BOOTSTRAP_REPLICATES = 2000
CALIBRATION_BOOTSTRAP_REPLICATES = 300
SEED = 707


def hours(value: str) -> float:
    hour, minute = map(int, value.split(":"))
    if not 0 <= minute < 60:
        raise ValueError(f"Invalid time: {value}")
    return hour + minute / 60


def source_records(set_name: str):
    archive = DATA / f"{set_name}.tar.gz"
    directory = DATA / set_name
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            members = sorted(
                (
                    member
                    for member in bundle
                    if member.isfile()
                    and member.name.startswith(f"{set_name}/")
                    and member.name.endswith(".txt")
                ),
                key=lambda member: member.name,
            )
            if len(members) != 4000:
                raise ValueError(f"{archive} has {len(members)} records, expected 4000")
            for member in members:
                with bundle.extractfile(member) as binary:
                    yield Path(member.name).stem, io.TextIOWrapper(binary, encoding="utf-8")
    elif directory.is_dir():
        files = sorted(directory.glob("*.txt"))
        if len(files) != 4000:
            raise ValueError(f"{directory} has {len(files)} records, expected 4000")
        for path in files:
            with path.open(encoding="utf-8", newline="") as source:
                yield path.stem, source
    else:
        raise FileNotFoundError(f"Missing {archive} or {directory}")


def build_feature_table(set_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    source_ids = []
    failures = []
    for record_id, source in source_records(set_name):
        try:
            numeric_id = int(record_id)
            source_ids.append(numeric_id)
            reader = csv.DictReader(source)
            if reader.fieldnames != ["Time", "Parameter", "Value"]:
                raise ValueError(f"Unexpected fields: {reader.fieldnames}")
            for row in reader:
                raw = row["Value"]
                value = np.nan if raw == "-1" else float(raw)
                rows.append((numeric_id, hours(row["Time"]), row["Parameter"], raw, value))
        except Exception as exc:
            failures.append((record_id, str(exc)))
    failure_table = pd.DataFrame(failures, columns=["file", "error"])
    failure_table.to_csv(OUT / f"cp3_{set_name}_read_failures.csv", index=False)
    if failures:
        raise ValueError(f"Read failures in {set_name}: {failures[:3]}")
    if len(source_ids) != len(set(source_ids)) or len(source_ids) != 4000:
        raise ValueError(f"Invalid RecordID count in {set_name}")

    long = pd.DataFrame(rows, columns=["RecordID", "time_hours", "Parameter", "value_raw", "value"])
    long, cleaning_masks = clean_measurements(long)
    minus_one = cleaning_masks["minus_one"]
    invalid = cleaning_masks["documented_invalid"]
    wide = build_modeling_table(long, source_ids)
    if len(wide) != 4000 or wide.RecordID.duplicated().any():
        raise ValueError(f"Invalid wide table for {set_name}")
    quality = pd.DataFrame(
        [
            {
                "set": set_name,
                "records": len(source_ids),
                "long_rows": len(long),
                "minus_one_values": int(minus_one.sum()),
                "invalid_values_set_missing": int(invalid.sum()),
                "times_outside_0_48": int((~long.time_hours.between(0, 48)).sum()),
            }
        ]
    )
    return wide, quality


def align_and_predict(wide: pd.DataFrame, bundle: dict) -> np.ndarray:
    if set(wide.columns) & OUTCOME_COLUMNS:
        raise AssertionError("Outcome columns entered the feature table")
    expected = bundle["predictors"]
    missing = sorted(set(expected) - set(wide.columns))
    if missing:
        raise ValueError(f"Missing locked predictors: {missing}")
    x = wide.reindex(columns=expected).copy()
    if "ICUType" in x:
        x["ICUType"] = x["ICUType"].map(
            lambda value: str(int(value)) if pd.notna(value) else np.nan
        )
    probabilities = bundle["fitted_model"].predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid predicted probabilities")
    return probabilities


def load_outcomes(set_name: str, record_ids: pd.Series) -> pd.DataFrame:
    path = DATA / f"Outcomes-{set_name[-1]}.txt"
    outcomes = pd.read_csv(path)
    if outcomes.RecordID.duplicated().any() or set(outcomes.RecordID) != set(record_ids):
        raise ValueError(f"Outcome IDs do not match {set_name}")
    if not outcomes[TARGET].isin([0, 1]).all():
        raise ValueError(f"Invalid outcomes in {path}")
    return pd.DataFrame({"RecordID": record_ids}).merge(
        outcomes, on="RecordID", how="left", validate="one_to_one"
    )


def calibration_intercept_slope(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    intercept_result = sm.GLM(
        y,
        np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=logit,
    ).fit()
    slope_result = sm.GLM(
        y,
        sm.add_constant(logit),
        family=sm.families.Binomial(),
    ).fit()
    return float(intercept_result.params[0]), float(slope_result.params[1])


def metric_values(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y, probability)
    brier = brier_score_loss(y, probability)
    null_brier = float(np.mean((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "deaths": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "brier_score": brier,
        "scaled_brier_score": 1 - brier / null_brier,
        "log_loss": log_loss(y, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": threshold,
        "sensitivity": recall_score(y, prediction, zero_division=0),
        "specificity": tn / (tn + fp),
        "positive_predictive_value": precision_score(y, prediction, zero_division=0),
        "negative_predictive_value": tn / (tn + fn),
        "accuracy": accuracy_score(y, prediction),
    }


def bootstrap_intervals(y: np.ndarray, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        sampled_y = y[index]
        sampled_probability = probability[index]
        values = metric_values(sampled_y, sampled_probability, threshold)
        for metric, value in values.items():
            if metric not in {"n", "deaths", "threshold"}:
                rows.append((replicate, metric, value))
    table = pd.DataFrame(rows, columns=["replicate", "metric", "value"])
    return table.groupby("metric").value.quantile([0.025, 0.975]).unstack().rename(
        columns={0.025: "lower_95", 0.975: "upper_95"}
    ).reset_index()


def bootstrap_auc_interval(y: np.ndarray, score: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        index = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        estimates.append(roc_auc_score(y[index], score[index]))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def save_precision_recall_plot(frame: pd.DataFrame) -> None:
    y = frame[TARGET].to_numpy(dtype=int)
    probability = frame.predicted_risk.to_numpy()
    precision, recall, _ = precision_recall_curve(y, probability)
    baseline = float(y.mean())
    average_precision = average_precision_score(y, probability)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color="#2166AC", linewidth=2, label=f"Model (AP {average_precision:.3f})")
    ax.axhline(
        baseline,
        color="#B2182B",
        linestyle="--",
        linewidth=1.5,
        label=f"Death-rate baseline ({baseline:.3f})",
    )
    ax.set(
        title="Precision–recall curve on set-b",
        xlabel="Recall (sensitivity)",
        ylabel="Precision (positive predictive value)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "cp3_set-b_precision_recall.png", dpi=180)
    plt.close(fig)


def save_discrimination_tables(evaluated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    score_columns = {
        "Elastic net logistic": "predicted_risk",
        "SAPS-I": "SAPS-I",
        "SOFA": "SOFA",
    }
    for set_index, (set_name, frame) in enumerate(evaluated.items()):
        y = frame[TARGET].to_numpy(dtype=int)
        for score_index, (model_name, column) in enumerate(score_columns.items()):
            score = frame[column].to_numpy(dtype=float)
            if not np.isfinite(score).all():
                raise ValueError(f"Missing {model_name} scores in {set_name}")
            lower, upper = bootstrap_auc_interval(
                y, score, SEED + 100 * set_index + score_index
            )
            rows.append(
                {
                    "dataset": set_name,
                    "model": model_name,
                    "auroc": roc_auc_score(y, score),
                    "lower_95": lower,
                    "upper_95": upper,
                }
            )
    external = pd.DataFrame(rows)
    external.to_csv(OUT / "cp3_discrimination.csv", index=False)

    cp2 = pd.read_csv(OUT / "cp2_model_comparison.csv")
    cp2_elastic = cp2.loc[cp2.model.eq("Elastic net logistic")].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "estimate": "Checkpoint 2 set-a nested CV",
                "auroc": cp2_elastic.mean_auroc,
                "uncertainty_type": "fold SD",
                "uncertainty": cp2_elastic.sd_auroc,
            },
            *[
                {
                    "estimate": f"{set_name} external validation",
                    "auroc": external.loc[
                        external.dataset.eq(set_name)
                        & external.model.eq("Elastic net logistic"),
                        "auroc",
                    ].iloc[0],
                    "uncertainty_type": "bootstrap 95% CI",
                    "uncertainty": (
                        f"{external.loc[external.dataset.eq(set_name) & external.model.eq('Elastic net logistic'), 'lower_95'].iloc[0]:.3f}–"
                        f"{external.loc[external.dataset.eq(set_name) & external.model.eq('Elastic net logistic'), 'upper_95'].iloc[0]:.3f}"
                    ),
                }
                for set_name in ["set-b", "set-c"]
            ],
        ]
    )
    comparison.to_csv(OUT / "cp3_auc_comparison.csv", index=False)

    rng = np.random.default_rng(SEED + 400)
    b = evaluated["set-b"]
    c = evaluated["set-c"]
    differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_scores = []
        for frame in [b, c]:
            y = frame[TARGET].to_numpy(dtype=int)
            positive = np.flatnonzero(y == 1)
            negative = np.flatnonzero(y == 0)
            index = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            sampled_scores.append(
                roc_auc_score(y[index], frame.predicted_risk.to_numpy()[index])
            )
        differences.append(sampled_scores[1] - sampled_scores[0])
    difference = pd.DataFrame(
        [
            {
                "contrast": "set-c minus set-b AUROC",
                "difference": (
                    external.loc[external.dataset.eq("set-c") & external.model.eq("Elastic net logistic"), "auroc"].iloc[0]
                    - external.loc[external.dataset.eq("set-b") & external.model.eq("Elastic net logistic"), "auroc"].iloc[0]
                ),
                "lower_95": np.quantile(differences, 0.025),
                "upper_95": np.quantile(differences, 0.975),
            }
        ]
    )
    difference.to_csv(OUT / "cp3_auc_difference.csv", index=False)
    return external


def net_benefit(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    true_positive = np.sum(prediction & (y == 1))
    false_positive = np.sum(prediction & (y == 0))
    return true_positive / len(y) - false_positive / len(y) * threshold / (1 - threshold)


def save_decision_curve_and_utility(
    set_a: pd.DataFrame,
    set_b: pd.DataFrame,
    locked_threshold: float,
) -> None:
    sofa_median = float(set_a["SOFA"].median())
    sofa_train = set_a["SOFA"].fillna(sofa_median).to_numpy(dtype=float)
    sofa_model = sm.GLM(
        set_a[TARGET].to_numpy(dtype=int),
        sm.add_constant(sofa_train),
        family=sm.families.Binomial(),
    ).fit()
    sofa_b = set_b["SOFA"].fillna(sofa_median).to_numpy(dtype=float)
    sofa_probability = np.asarray(sofa_model.predict(sm.add_constant(sofa_b)))
    y = set_b[TARGET].to_numpy(dtype=int)
    model_probability = set_b.predicted_risk.to_numpy(dtype=float)
    prevalence = float(y.mean())
    thresholds = np.linspace(0.05, 0.50, 46)
    rows = []
    for threshold in thresholds:
        rows.extend(
            [
                (threshold, "Elastic net model", net_benefit(y, model_probability, threshold)),
                (threshold, "SOFA", net_benefit(y, sofa_probability, threshold)),
                (
                    threshold,
                    "Treat all",
                    prevalence - (1 - prevalence) * threshold / (1 - threshold),
                ),
                (threshold, "Treat none", 0.0),
            ]
        )
    decision = pd.DataFrame(rows, columns=["threshold", "strategy", "net_benefit"])
    decision.to_csv(OUT / "cp3_decision_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = {
        "Elastic net model": "#2166AC",
        "SOFA": "#D6604D",
        "Treat all": "#777777",
        "Treat none": "#000000",
    }
    styles = {"Elastic net model": "-", "SOFA": "-", "Treat all": "--", "Treat none": ":"}
    for strategy, group in decision.groupby("strategy"):
        ax.plot(
            group.threshold,
            group.net_benefit,
            color=colors[strategy],
            linestyle=styles[strategy],
            linewidth=2,
            label=strategy,
        )
    ax.axvline(locked_threshold, color="#4D4D4D", linestyle="-.", linewidth=1, label=f"Chosen threshold ({locked_threshold:.2f})")
    ax.set(
        title="Decision curve analysis on set-b",
        xlabel="Risk threshold",
        ylabel="Net benefit",
        xlim=(0.05, 0.50),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cp3_decision_curve.png", dpi=180)
    plt.close(fig)

    prediction = model_probability >= locked_threshold
    sensitivity = recall_score(y, prediction)
    ppv = precision_score(y, prediction, zero_division=0)
    pd.DataFrame(
        [
            {
                "proposed_use": "Flag for clinician review and possible goals-of-care conversation",
                "dataset": "set-b",
                "threshold": locked_threshold,
                "sensitivity": sensitivity,
                "positive_predictive_value": ppv,
                "flagged_per_100": 100 * prediction.mean(),
                "challenge_event_1_score": min(sensitivity, ppv),
            }
        ]
    ).to_csv(OUT / "cp3_clinical_utility.csv", index=False)


def prepare_locked_predictors(wide: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    x = wide.reindex(columns=bundle["predictors"]).copy()
    if "ICUType" in x:
        x["ICUType"] = x["ICUType"].map(
            lambda value: str(int(value)) if pd.notna(value) else np.nan
        )
    return x


def save_interpretation_and_mistakes(
    bundle: dict,
    wide_b: pd.DataFrame,
    evaluated_b: pd.DataFrame,
) -> None:
    pipeline = bundle["fitted_model"]
    preprocess = pipeline.named_steps["preprocess"]
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    x = prepare_locked_predictors(wide_b, bundle)
    transformed = preprocess.transform(x)
    standardized = scaler.transform(transformed)
    names = preprocess.get_feature_names_out()
    coefficients = model.coef_.ravel()
    coefficient_table = pd.DataFrame(
        {
            "feature": names,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.to_csv(OUT / "cp3_model_coefficients.csv", index=False)

    indexed_evaluation = evaluated_b.set_index("RecordID")
    high_risk_survivors = (
        evaluated_b.loc[evaluated_b[TARGET].eq(0)]
        .nlargest(5, "predicted_risk")
        .assign(error_type="Highest predicted risk among survivors")
    )
    low_risk_deaths = (
        evaluated_b.loc[evaluated_b[TARGET].eq(1)]
        .nsmallest(5, "predicted_risk")
        .assign(error_type="Lowest predicted risk among deaths")
    )
    selected = pd.concat([high_risk_survivors, low_risk_deaths], ignore_index=True)
    row_lookup = pd.Series(np.arange(len(wide_b)), index=wide_b.RecordID)
    missing_fraction = wide_b[bundle["predictors"]].isna().mean(axis=1).to_numpy()
    rows = []
    for record in selected.itertuples(index=False):
        row_index = int(row_lookup.loc[record.RecordID])
        contributions = standardized[row_index] * coefficients
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        driver_text = "; ".join(
            f"{names[index]} ({contributions[index]:+.2f} log-odds)"
            for index in top_indices
        )
        top_names = [str(names[index]) for index in top_indices]
        fraction = float(missing_fraction[row_index])
        if fraction >= 0.50:
            judgment = "Data limitation: heavy predictor missingness likely contributed"
        elif any("missingindicator" in name or name.endswith("_count") for name in top_names):
            judgment = "Model/data-process limitation: missingness or measurement intensity strongly influenced risk"
        else:
            judgment = "Information/model limitation: relevant post-48-hour or unmeasured factors may be absent"
        rows.append(
            {
                "RecordID": int(record.RecordID),
                "error_type": record.error_type,
                "observed_death": int(indexed_evaluation.loc[record.RecordID, TARGET]),
                "predicted_risk": float(indexed_evaluation.loc[record.RecordID, "predicted_risk"]),
                "missing_predictor_fraction": fraction,
                "top_feature_contributions": driver_text,
                "judgment": judgment,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "cp3_confident_mistakes.csv", index=False)


def save_validation_plots(evaluated: dict[str, pd.DataFrame]) -> None:
    palette = {"set-b": "#2166AC", "set-c": "#B2182B"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for set_name, frame in evaluated.items():
        fpr, tpr, _ = roc_curve(frame[TARGET], frame.predicted_risk)
        score = roc_auc_score(frame[TARGET], frame.predicted_risk)
        ax.plot(fpr, tpr, color=palette[set_name], linewidth=2, label=f"{set_name} (AUROC {score:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set(title="Locked-model external validation", xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "cp3_external_roc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex="col",
    )
    for column_index, (set_name, frame) in enumerate(evaluated.items()):
        ax = axes[0, column_index]
        hist_ax = axes[1, column_index]
        y = frame[TARGET].to_numpy(dtype=int)
        probability = frame.predicted_risk.to_numpy(dtype=float)
        upper = max(0.5, float(np.quantile(probability, 0.995)))
        grid = np.linspace(0.001, upper, 150)
        fitted = lowess(y, probability, frac=0.35, it=0, return_sorted=True)
        estimate = np.interp(grid, fitted[:, 0], fitted[:, 1])

        rng = np.random.default_rng(SEED + 500 + column_index)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        bootstrap_curves = []
        for _ in range(CALIBRATION_BOOTSTRAP_REPLICATES):
            index = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            fitted_bootstrap = lowess(
                y[index], probability[index], frac=0.35, it=0, return_sorted=True
            )
            bootstrap_curves.append(
                np.interp(grid, fitted_bootstrap[:, 0], fitted_bootstrap[:, 1])
            )
        lower, upper_band = np.quantile(bootstrap_curves, [0.025, 0.975], axis=0)
        ax.plot(grid, estimate, color=palette[set_name], linewidth=2, label="LOESS estimate")
        ax.fill_between(grid, lower, upper_band, color=palette[set_name], alpha=0.20, label="95% bootstrap band")
        ax.plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=1, label="Ideal")
        ax.set(
            title=f"Flexible calibration: {set_name}",
            ylabel="Observed mortality proportion",
            xlim=(0, upper),
            ylim=(0, min(1, upper + 0.08)),
        )
        ax.legend(loc="upper left")
        hist_ax.hist(probability, bins=35, range=(0, upper), color=palette[set_name], alpha=0.75)
        hist_ax.set(xlabel="Predicted mortality risk", ylabel="Count", xlim=(0, upper))
    fig.tight_layout()
    fig.savefig(FIG / "cp3_calibration.png", dpi=180)
    plt.close(fig)

    combined = pd.concat(
        [frame.assign(dataset=set_name) for set_name, frame in evaluated.items()], ignore_index=True
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=combined, x="dataset", y="predicted_risk", hue=TARGET, ax=ax)
    ax.set(title="Predicted risk by outcome and validation set", xlabel="", ylabel="Predicted mortality risk")
    ax.legend(title="In-hospital death")
    fig.tight_layout()
    fig.savefig(FIG / "cp3_risk_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    bundle = joblib.load(OUT / "cp2_final_model.joblib")
    if bundle["model_name"] != "Elastic net logistic":
        raise ValueError(f"Unexpected locked model: {bundle['model_name']}")

    oof = pd.read_csv(OUT / "cp2_oof_predictions.csv")
    threshold_grid = np.linspace(0.01, 0.99, 99)
    threshold_scores = [
        min(
            recall_score(oof.outcome, oof["Elastic net logistic"] >= threshold),
            precision_score(oof.outcome, oof["Elastic net logistic"] >= threshold, zero_division=0),
        )
        for threshold in threshold_grid
    ]
    locked_threshold = float(threshold_grid[int(np.argmax(threshold_scores))])

    predictions = {}
    feature_tables = {}
    quality_rows = []
    for set_name in ["set-b", "set-c"]:
        wide, quality = build_feature_table(set_name)
        feature_tables[set_name] = wide
        quality_rows.append(quality)
        probability = align_and_predict(wide, bundle)
        predictions[set_name] = pd.DataFrame(
            {
                "RecordID": wide.RecordID.astype(int),
                "predicted_death": (probability >= locked_threshold).astype(int),
                "predicted_risk": probability,
            }
        )
        predictions[set_name].to_csv(OUT / f"cp3_{set_name}_predictions.csv", index=False)

    # Outcomes are deliberately loaded only after both locked prediction files exist.
    evaluated = {}
    for set_name in ["set-b", "set-c"]:
        outcomes = load_outcomes(set_name, predictions[set_name].RecordID)
        evaluated[set_name] = predictions[set_name].merge(
            outcomes[["RecordID", TARGET, "SAPS-I", "SOFA"]],
            on="RecordID",
            validate="one_to_one",
        )

    discrimination = save_discrimination_tables(evaluated)
    save_precision_recall_plot(evaluated["set-b"])
    set_a = pd.read_csv(OUT / "set-a_wide.csv")
    save_decision_curve_and_utility(set_a, evaluated["set-b"], locked_threshold)
    save_interpretation_and_mistakes(
        bundle,
        feature_tables["set-b"],
        evaluated["set-b"],
    )

    metric_rows = []
    interval_rows = []
    subgroup_rows = []
    for set_name, frame in evaluated.items():
        y = frame[TARGET].to_numpy(dtype=int)
        probability = frame.predicted_risk.to_numpy()
        values = metric_values(y, probability, locked_threshold)
        metric_rows.append({"set": set_name, **values})
        intervals = bootstrap_intervals(y, probability, locked_threshold)
        intervals.insert(0, "set", set_name)
        interval_rows.append(intervals)

        if set_name == "set-b":
            audit = frame.merge(
                feature_tables[set_name][["RecordID", "Age", "Gender", "ICUType"]],
                on="RecordID",
                validate="one_to_one",
            )
            audit["Age group"] = pd.cut(
                audit.Age,
                bins=[-np.inf, 64, 79, np.inf],
                labels=["<65", "65–79", "≥80"],
            )
            audit["Sex"] = audit.Gender.map({0: "Female", 1: "Male"}).fillna("Missing")
            audit["ICU type"] = audit.ICUType.map(
                {
                    1: "Coronary care",
                    2: "Cardiac surgery",
                    3: "Medical ICU",
                    4: "Surgical ICU",
                }
            ).fillna("Missing")
            for variable in ["Age group", "Sex", "ICU type"]:
                for level, group in audit.groupby(variable, observed=False, dropna=False):
                    if len(group) == 0 or group[TARGET].nunique() < 2:
                        continue
                    subgroup_rows.append(
                        {
                            "set": set_name,
                            "subgroup_variable": variable,
                            "subgroup_level": "Missing" if pd.isna(level) else str(level),
                            **metric_values(
                                group[TARGET].to_numpy(dtype=int),
                                group.predicted_risk.to_numpy(),
                                locked_threshold,
                            ),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "cp3_validation_metrics.csv", index=False)
    pd.concat(interval_rows, ignore_index=True).to_csv(
        OUT / "cp3_validation_intervals.csv", index=False
    )
    pd.DataFrame(subgroup_rows).to_csv(OUT / "cp3_subgroup_audit.csv", index=False)
    pd.concat(quality_rows, ignore_index=True).to_csv(OUT / "cp3_data_quality.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": bundle["model_name"],
                "training_set": "set-a",
                "validation_sets": "set-b; set-c",
                "threshold": locked_threshold,
                "threshold_source": "set-a out-of-fold predictions only",
                "model_parameters": json.dumps(bundle["parameters"]),
                "outcome_access_rule": "Outcomes-b/c loaded only after locked predictions were saved",
            }
        ]
    ).to_csv(OUT / "cp3_validation_protocol.csv", index=False)
    save_validation_plots(evaluated)
    print(
        "Completed locked external validation; "
        + "; ".join(
            f"{row['set']} AUROC={row['auroc']:.3f}, Brier={row['brier_score']:.3f}"
            for row in metric_rows
        )
    )


if __name__ == "__main__":
    main()
