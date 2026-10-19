"""Shared cleaning and modeling-table construction for PhysioNet sets A, B, and C."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_NAMES = ["Age", "Gender", "Height", "ICUType", "Weight"]


def clean_measurements(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    cleaned = long.copy()
    raw_numeric = pd.to_numeric(cleaned["value_raw"], errors="coerce")
    minus_one = cleaned.value_raw.eq("-1")
    negative_invalid = raw_numeric.lt(0) & ~minus_one
    ph_invalid = cleaned.Parameter.eq("pH") & ~raw_numeric.between(0, 14) & ~minus_one
    gcs_invalid = cleaned.Parameter.eq("GCS") & ~raw_numeric.between(3, 15) & ~minus_one
    fio2_invalid = cleaned.Parameter.eq("FiO2") & ~raw_numeric.between(0, 1) & ~minus_one
    gender_invalid = cleaned.Parameter.eq("Gender") & ~raw_numeric.isin([0, 1]) & ~minus_one
    icu_invalid = cleaned.Parameter.eq("ICUType") & ~raw_numeric.isin([1, 2, 3, 4]) & ~minus_one
    vent_invalid = cleaned.Parameter.eq("MechVent") & ~raw_numeric.isin([0, 1]) & ~minus_one
    zero_height_weight = cleaned.Parameter.isin(["Height", "Weight"]) & raw_numeric.eq(0)
    documented_invalid = (
        negative_invalid
        | ph_invalid
        | gcs_invalid
        | fio2_invalid
        | gender_invalid
        | icu_invalid
        | vent_invalid
        | zero_height_weight
    )
    cleaned.loc[documented_invalid, "value"] = np.nan
    masks = {
        "raw_numeric": raw_numeric,
        "minus_one": minus_one,
        "negative_invalid": negative_invalid,
        "ph_invalid": ph_invalid,
        "gcs_invalid": gcs_invalid,
        "fio2_invalid": fio2_invalid,
        "gender_invalid": gender_invalid,
        "icu_invalid": icu_invalid,
        "vent_invalid": vent_invalid,
        "zero_height_weight": zero_height_weight,
        "documented_invalid": documented_invalid,
    }
    return cleaned, masks


def build_modeling_table(long: pd.DataFrame, source_ids: list[int]) -> pd.DataFrame:
    static = (
        long[long.Parameter.isin(STATIC_NAMES)]
        .sort_values(["RecordID", "Parameter", "time_hours"])
        .drop_duplicates(["RecordID", "Parameter"])
        .pivot(index="RecordID", columns="Parameter", values="value")
        .reindex(source_ids)
        .rename_axis("RecordID")
        .reset_index()
    )
    dynamic = long[~long.Parameter.isin(STATIC_NAMES + ["RecordID"])].sort_values(
        ["RecordID", "Parameter", "time_hours"]
    )
    summary = dynamic.groupby(["RecordID", "Parameter"])["value"].agg(
        ["count", "first", "last", "min", "max", "mean"]
    )
    features = summary.unstack("Parameter")
    features.columns = [f"{parameter}_{stat}" for stat, parameter in features.columns]
    for parameter in dynamic.Parameter.unique():
        features[f"{parameter}_count"] = features[f"{parameter}_count"].fillna(0).astype(int)
    wide = static.merge(features.reset_index(), on="RecordID", how="left", validate="one_to_one")
    if len(wide) != len(source_ids) or wide.RecordID.duplicated().any():
        raise ValueError("Invalid modeling table")
    return wide
