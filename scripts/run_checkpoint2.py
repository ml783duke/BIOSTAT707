"""Build, evaluate, and save BIOSTAT 707 Checkpoint 2 models."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
import seaborn as sns
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message="overflow encountered in exp",
    category=RuntimeWarning,
    module="statsmodels.genmod.families.links",
)


ROOT = Path.cwd()
OUT = ROOT / "output"
FIG = OUT / "figures"
SEED = 707
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
TARGET = "In-hospital_death"
BENCHMARKS = ["SAPS-I", "SOFA"]
EXCLUDED = {"RecordID", TARGET, "Length_of_stay", "Survival", *BENCHMARKS}

SPLINE_CONTINUOUS = [
    "Age",
    "Weight",
    "HR_last",
    "Temp_last",
    "RespRate_last",
    "MAP_last",
    "SysABP_last",
    "BUN_last",
    "Creatinine_last",
    "WBC_last",
    "Platelets_last",
    "Glucose_last",
    "GCS_last",
]
SPLINE_CATEGORICAL = ["Gender", "ICUType"]
EFFECT_VARIABLES = ["Age", "HR_last", "Temp_last"]


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold: float = 0.90, scale: bool = False, drop_first: bool = False):
        self.missing_threshold = missing_threshold
        self.scale = scale
        self.drop_first = drop_first

    def fit(self, x: pd.DataFrame, y=None):
        missing_proportion = x.isna().mean()
        self.retained_columns_ = missing_proportion[missing_proportion <= self.missing_threshold].index.tolist()
        if "ICUType" not in self.retained_columns_:
            raise ValueError("ICUType was removed by the missingness screen.")
        numeric_columns = [column for column in self.retained_columns_ if column != "ICUType"]
        numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if self.scale:
            numeric_steps.append(("scale", StandardScaler()))
        numeric = Pipeline(numeric_steps)
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first" if self.drop_first else None,
                    ),
                ),
            ]
        )
        self.transformer_ = ColumnTransformer(
            [("numeric", numeric, numeric_columns), ("icu", categorical, ["ICUType"])],
            verbose_feature_names_out=False,
            sparse_threshold=0.0,
        )
        self.transformer_.fit(x[self.retained_columns_], y)
        return self

    def transform(self, x: pd.DataFrame):
        return self.transformer_.transform(x[self.retained_columns_])

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out()


def make_elastic_net_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", FoldPreprocessor(drop_first=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=3000,
                    tol=1e-3,
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", FoldPreprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


class NaturalSplinePreprocessor:
    def fit(self, x: pd.DataFrame):
        self.continuous_medians_ = {
            column: float(x[column].median()) for column in SPLINE_CONTINUOUS
        }
        self.categorical_modes_ = {
            column: x[column].mode(dropna=True).iloc[0] for column in SPLINE_CATEGORICAL
        }
        self.categorical_levels_ = {
            column: sorted(x[column].dropna().unique().tolist()) for column in SPLINE_CATEGORICAL
        }
        frame = self._prepare(x)
        terms = [f"cr({column}, df=4)" for column in SPLINE_CONTINUOUS]
        terms.extend(f"C({column})" for column in SPLINE_CATEGORICAL)
        matrix = dmatrix(" + ".join(terms), frame, return_type="dataframe")
        self.design_info_ = matrix.design_info
        return self

    def _prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x[SPLINE_CONTINUOUS + SPLINE_CATEGORICAL].copy()
        for column, median in self.continuous_medians_.items():
            frame[column] = frame[column].fillna(median)
        for column, mode in self.categorical_modes_.items():
            levels = self.categorical_levels_[column]
            frame[column] = frame[column].fillna(mode)
            frame.loc[~frame[column].isin(levels), column] = mode
            frame[column] = pd.Categorical(frame[column], categories=levels)
        return frame

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        frame = self._prepare(x)
        matrix = build_design_matrices([self.design_info_], frame)[0]
        return np.asarray(matrix)


def fit_spline_model(train_x: pd.DataFrame, train_y: pd.Series):
    preprocess = NaturalSplinePreprocessor().fit(train_x)
    design = preprocess.transform(train_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.GLM(train_y.to_numpy(), design, family=sm.families.Binomial()).fit()
    return preprocess, result


def predict_spline(preprocess, result, frame: pd.DataFrame) -> np.ndarray:
    design = preprocess.transform(frame)
    return np.asarray(result.predict(design))


def choose_final_model(summary: pd.DataFrame) -> str:
    modeled = summary[summary["model"].isin(["Elastic net logistic", "Spline logistic", "Random forest"])].copy()
    best_auc = modeled["mean_auroc"].max()
    elastic_net_auc = modeled.loc[modeled.model.eq("Elastic net logistic"), "mean_auroc"].iloc[0]
    if best_auc - elastic_net_auc <= 0.01:
        return "Elastic net logistic"
    return modeled.sort_values(["mean_auroc", "model"], ascending=[False, True]).iloc[0]["model"]


def save_coefficient_plot(fitted: Pipeline) -> pd.DataFrame:
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    coefficients = fitted.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["abs_coefficient"] = table.coefficient.abs()
    table["selected"] = table.abs_coefficient.gt(1e-8)
    table.to_csv(OUT / "cp2_elastic_net_coefficients.csv", index=False)
    shown = table.nlargest(25, "abs_coefficient").sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = np.where(shown.coefficient >= 0, "#B2182B", "#2166AC")
    ax.barh(shown.feature, shown.coefficient, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Largest standardized elastic net coefficients", xlabel="Coefficient (log-odds scale)", ylabel="")
    fig.tight_layout()
    fig.savefig(FIG / "cp2_elastic_net_coefficients.png", dpi=180)
    plt.close(fig)
    return table


def save_effect_plots(data_x: pd.DataFrame, preprocess, result) -> pd.DataFrame:
    reference = {}
    for column in SPLINE_CONTINUOUS:
        reference[column] = float(data_x[column].median())
    for column in SPLINE_CATEGORICAL:
        reference[column] = data_x[column].mode(dropna=True).iloc[0]
    reference_frame = pd.DataFrame([reference])
    reference_design = preprocess.transform(reference_frame)[0]
    covariance = np.asarray(result.cov_params())
    rows = []
    fig, axes = plt.subplots(1, len(EFFECT_VARIABLES), figsize=(15, 4.5))
    for ax, variable in zip(axes, EFFECT_VARIABLES):
        observed = data_x[variable].dropna()
        lower, upper = observed.quantile([0.02, 0.98])
        grid = np.linspace(lower, upper, 120)
        frames = pd.concat([reference_frame] * len(grid), ignore_index=True)
        frames[variable] = grid
        design = preprocess.transform(frames)
        contrast = design - reference_design
        log_or = contrast @ np.asarray(result.params)
        variance = np.einsum("ij,jk,ik->i", contrast, covariance, contrast)
        standard_error = np.sqrt(np.maximum(variance, 0))
        odds_ratio = np.exp(log_or)
        low = np.exp(log_or - 1.96 * standard_error)
        high = np.exp(log_or + 1.96 * standard_error)
        for value, estimate, lo, hi in zip(grid, odds_ratio, low, high):
            rows.append((variable, value, estimate, lo, hi, reference[variable]))
        ax.plot(grid, odds_ratio, color="#2166AC", linewidth=2)
        ax.fill_between(grid, low, high, color="#92C5DE", alpha=0.45)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(
            title=variable,
            xlabel=f"{variable} (reference {reference[variable]:.1f})",
            ylabel="Odds ratio" if variable == EFFECT_VARIABLES[0] else "",
        )
    fig.suptitle("Adjusted spline effects with 95% confidence bands", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "cp2_spline_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    table = pd.DataFrame(rows, columns=["feature", "value", "odds_ratio", "lower_95", "upper_95", "reference"])
    table.to_csv(OUT / "cp2_spline_effects.csv", index=False)
    return table


def save_roc_plot(y: pd.Series, predictions: pd.DataFrame) -> None:
    labels = ["Elastic net logistic", "Spline logistic", "Random forest", "SAPS-I", "SOFA"]
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("colorblind", len(labels))
    for label, color in zip(labels, palette):
        fpr, tpr, _ = roc_curve(y, predictions[label])
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{label} (AUROC {score:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set(
        title="Out-of-fold ROC curves on set-a",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "cp2_oof_roc.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    data = pd.read_csv(OUT / "set-a_wide.csv")
    assert len(data) == data.RecordID.nunique() == 4000
    assert data[TARGET].isin([0, 1]).all()
    assert not any(name in EXCLUDED - {"RecordID"} for name in [])

    candidate_columns = [column for column in data.columns if column not in EXCLUDED]
    rare_columns = [column for column in candidate_columns if data[column].isna().mean() > 0.90]
    retained = [column for column in candidate_columns if column not in rare_columns]
    assert "ICUType" in retained
    assert set(SPLINE_CONTINUOUS + SPLINE_CATEGORICAL).issubset(data.columns)

    x = data[candidate_columns].copy()
    x["ICUType"] = x["ICUType"].map(
        lambda value: str(int(value)) if pd.notna(value) else np.nan
    )
    spline_x = data[SPLINE_CONTINUOUS + SPLINE_CATEGORICAL].copy()
    for column in SPLINE_CATEGORICAL:
        spline_x[column] = spline_x[column].map(
            lambda value: str(int(value)) if pd.notna(value) else np.nan
        )
    y = data[TARGET].astype(int)

    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.zeros(len(data), dtype=int)
    oof = pd.DataFrame(index=data.index)
    tuning_rows = []
    elastic_stability_rows = []

    for fold, (train_index, test_index) in enumerate(outer.split(x, y), start=1):
        fold_id[test_index] = fold
        inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED + fold)

        elastic_search = GridSearchCV(
            make_elastic_net_pipeline(),
            {
                "model__C": [0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            scoring="neg_log_loss",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        elastic_search.fit(x.iloc[train_index], y.iloc[train_index])
        oof.loc[test_index, "Elastic net logistic"] = elastic_search.predict_proba(x.iloc[test_index])[:, 1]
        elastic_n_predictors = len(elastic_search.best_estimator_.named_steps["preprocess"].retained_columns_)
        tuning_rows.append(
            (fold, "Elastic net logistic", json.dumps(elastic_search.best_params_), -elastic_search.best_score_, elastic_n_predictors)
        )
        elastic_names = elastic_search.best_estimator_.named_steps["preprocess"].get_feature_names_out()
        elastic_coefficients = elastic_search.best_estimator_.named_steps["model"].coef_.ravel()
        for feature, coefficient in zip(elastic_names, elastic_coefficients):
            elastic_stability_rows.append((fold, feature, coefficient, abs(coefficient) > 1e-8))

        spline_preprocess, spline_result = fit_spline_model(spline_x.iloc[train_index], y.iloc[train_index])
        oof.loc[test_index, "Spline logistic"] = predict_spline(
            spline_preprocess, spline_result, spline_x.iloc[test_index]
        )

        forest_search = GridSearchCV(
            make_forest_pipeline(),
            {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [5, 15],
                "model__max_features": ["sqrt", 0.4],
            },
            scoring="neg_log_loss",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        forest_search.fit(x.iloc[train_index], y.iloc[train_index])
        oof.loc[test_index, "Random forest"] = forest_search.predict_proba(x.iloc[test_index])[:, 1]
        forest_n_predictors = len(forest_search.best_estimator_.named_steps["preprocess"].retained_columns_)
        tuning_rows.append(
            (fold, "Random forest", json.dumps(forest_search.best_params_), -forest_search.best_score_, forest_n_predictors)
        )

        for benchmark in BENCHMARKS:
            train_median = data.loc[train_index, benchmark].median()
            oof.loc[test_index, benchmark] = data.loc[test_index, benchmark].fillna(train_median)

    assert fold_id.min() == 1 and fold_id.max() == N_OUTER_FOLDS
    assert not oof.isna().any().any()

    comparison_rows = []
    for model in oof.columns:
        fold_scores = []
        for fold in range(1, N_OUTER_FOLDS + 1):
            selected = fold_id == fold
            score = roc_auc_score(y[selected], oof.loc[selected, model])
            fold_scores.append(score)
            comparison_rows.append((model, fold, score))
        comparison_rows.append((model, "Mean", float(np.mean(fold_scores))))
        comparison_rows.append((model, "SD", float(np.std(fold_scores, ddof=1))))
    long_comparison = pd.DataFrame(comparison_rows, columns=["model", "fold", "auroc"])
    long_comparison.to_csv(OUT / "cp2_fold_auroc_long.csv", index=False)
    wide_comparison = long_comparison.pivot(index="model", columns="fold", values="auroc").reset_index()
    wide_comparison.columns = [str(column).lower().replace(" ", "_") for column in wide_comparison.columns]
    ordered = ["model", "1", "2", "3", "4", "5", "mean", "sd"]
    wide_comparison = wide_comparison[ordered].rename(
        columns={str(index): f"fold_{index}" for index in range(1, 6)}
    )
    wide_comparison = wide_comparison.rename(columns={"mean": "mean_auroc", "sd": "sd_auroc"})
    wide_comparison.to_csv(OUT / "cp2_model_comparison.csv", index=False)

    predictions = pd.DataFrame({"RecordID": data.RecordID, "fold": fold_id, "outcome": y})
    for model in oof.columns:
        predictions[model] = oof[model]
    predictions.to_csv(OUT / "cp2_oof_predictions.csv", index=False)
    pd.DataFrame(
        tuning_rows,
        columns=["outer_fold", "model", "best_parameters", "inner_log_loss", "outer_training_predictors"],
    ).to_csv(
        OUT / "cp2_tuning_results.csv", index=False
    )
    stability = pd.DataFrame(
        elastic_stability_rows,
        columns=["outer_fold", "feature", "coefficient", "selected"],
    )
    stability.to_csv(OUT / "cp2_elastic_net_stability_long.csv", index=False)
    stability.groupby("feature").agg(
        selected_folds=("selected", "sum"),
        selection_rate=("selected", "mean"),
        mean_coefficient=("coefficient", "mean"),
    ).reset_index().sort_values(
        ["selected_folds", "feature"], ascending=[False, True]
    ).to_csv(OUT / "cp2_elastic_net_stability.csv", index=False)
    pd.DataFrame(
        {
            "predictor": candidate_columns,
            "missing_proportion": [data[column].isna().mean() for column in candidate_columns],
            "status": ["Dropped: >90% missing" if column in rare_columns else "Retained" for column in candidate_columns],
        }
    ).to_csv(OUT / "cp2_predictor_screening.csv", index=False)

    final_inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
    elastic_final_search = GridSearchCV(
        make_elastic_net_pipeline(),
        {
            "model__C": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        scoring="neg_log_loss",
        cv=final_inner,
        n_jobs=-1,
        refit=True,
    )
    elastic_final_search.fit(x, y)
    save_coefficient_plot(elastic_final_search.best_estimator_)

    spline_preprocess, spline_result = fit_spline_model(spline_x, y)
    save_effect_plots(spline_x, spline_preprocess, spline_result)

    forest_final_search = GridSearchCV(
        make_forest_pipeline(),
        {
            "model__max_depth": [6, None],
            "model__min_samples_leaf": [5, 15],
            "model__max_features": ["sqrt", 0.4],
        },
        scoring="neg_log_loss",
        cv=final_inner,
        n_jobs=-1,
        refit=True,
    )
    forest_final_search.fit(x, y)

    final_name = choose_final_model(wide_comparison)
    if final_name == "Elastic net logistic":
        final_object = elastic_final_search.best_estimator_
        final_parameters = elastic_final_search.best_params_
    elif final_name == "Random forest":
        final_object = forest_final_search.best_estimator_
        final_parameters = forest_final_search.best_params_
    else:
        final_object = {"preprocess": spline_preprocess, "model": spline_result}
        final_parameters = {"spline": "natural cubic", "degrees_of_freedom": 4}
    joblib.dump(
        {
            "model_name": final_name,
            "fitted_model": final_object,
            "predictors": (
                final_object.named_steps["preprocess"].retained_columns_
                if final_name != "Spline logistic"
                else SPLINE_CONTINUOUS + SPLINE_CATEGORICAL
            ),
            "parameters": final_parameters,
            "target": TARGET,
            "random_seed": SEED,
        },
        OUT / "cp2_final_model.joblib",
    )
    pd.DataFrame(
        [{"final_model": final_name, "parameters": json.dumps(final_parameters), "selection_rule": "Highest mean outer-CV AUROC; prefer elastic net when within 0.01 of the best."}]
    ).to_csv(OUT / "cp2_final_model_summary.csv", index=False)
    save_roc_plot(y, oof)
    print(
        f"Completed CP2 for {len(data)} stays with {int(y.sum())} deaths; "
        f"retained {len(retained)} predictors and selected {final_name}."
    )


if __name__ == "__main__":
    main()
