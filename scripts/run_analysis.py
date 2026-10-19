"""Generate Checkpoint 1 outputs from read-only set-a data."""
from pathlib import Path
import csv
import io
import tarfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from modeling_table import STATIC_NAMES, build_modeling_table, clean_measurements

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sns.set_theme(style="white")

def source_records():
    archive = DATA / "set-a.tar.gz"
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            members = sorted((m for m in bundle if m.isfile() and m.name.startswith("set-a/") and m.name.endswith(".txt")), key=lambda m: m.name)
            assert len(members) == 4000, f"Archive has {len(members)} records"
            for member in members:
                with bundle.extractfile(member) as binary:
                    yield Path(member.name).stem, io.TextIOWrapper(binary, encoding="utf-8")
    else:
        files = sorted((DATA / "set-a").glob("*.txt"))
        assert len(files) == 4000, f"Directory has {len(files)} records"
        for path in files:
            with path.open(encoding="utf-8", newline="") as source:
                yield path.stem, source

def hours(time):
    h, m = map(int, time.split(":"))
    assert 0 <= m < 60
    return h + m / 60

rows, source_ids, failures = [], [], []
for record_id, source in source_records():
    try:
        source_ids.append(int(record_id))
        reader = csv.DictReader(source)
        assert reader.fieldnames == ["Time", "Parameter", "Value"]
        record_rows = []
        for row in reader:
            raw = row["Value"]
            record_rows.append((int(record_id), row["Time"], hours(row["Time"]), row["Parameter"], raw, np.nan if raw == "-1" else float(raw)))
        rows.extend(record_rows)
    except Exception as exc:
        failures.append((record_id, str(exc)))
pd.DataFrame(failures, columns=["file", "error"]).to_csv(OUT / "read_failures.csv", index=False)
assert not failures, f"Read failures: {failures[:3]}"
assert len(source_ids) == len(set(source_ids)) == 4000
long = pd.DataFrame(rows, columns=["RecordID", "Time", "time_hours", "Parameter", "value_raw", "value"])
long, cleaning_masks = clean_measurements(long)
raw_numeric = cleaning_masks["raw_numeric"]
minus_one = cleaning_masks["minus_one"]
negative_invalid = cleaning_masks["negative_invalid"]
ph_invalid = cleaning_masks["ph_invalid"]
gcs_invalid = cleaning_masks["gcs_invalid"]
fio2_invalid = cleaning_masks["fio2_invalid"]
gender_invalid = cleaning_masks["gender_invalid"]
icu_invalid = cleaning_masks["icu_invalid"]
vent_invalid = cleaning_masks["vent_invalid"]
zero_height_weight = cleaning_masks["zero_height_weight"]
documented_invalid = cleaning_masks["documented_invalid"]
long.to_csv(OUT / "set-a_long.csv", index=False)
outcomes_raw = pd.read_csv(DATA / "Outcomes-a.txt")
assert not outcomes_raw.RecordID.duplicated().any()
assert set(outcomes_raw.RecordID) == set(source_ids)
outcomes = outcomes_raw.copy()
outcomes["Survival"] = outcomes.Survival.replace(-1, np.nan)

duplicate = long.duplicated(["RecordID", "Time", "Parameter"], keep=False)
quality = pd.DataFrame([
    ("Files read", len(source_ids)), ("Unique RecordIDs", long.RecordID.nunique()),
    ("Long rows", len(long)), ("Raw -1 codes", int(minus_one.sum())),
    ("Other negative values", int(negative_invalid.sum())),
    ("pH values outside documented 0-14 range", int(ph_invalid.sum())),
    ("Other documented-range violations", int((gcs_invalid | fio2_invalid | gender_invalid | icu_invalid | vent_invalid).sum())),
    ("Zero Height or Weight values", int(zero_height_weight.sum())),
    ("Times outside 0-48 hours", int((~long.time_hours.between(0, 48)).sum())),
    ("Rows in duplicate ID-variable-time groups", int(duplicate.sum())),
    ("Unmatched outcome IDs", int((~outcomes_raw.RecordID.isin(source_ids)).sum())),
], columns=["check", "count"])
quality.to_csv(OUT / "data_quality_summary.csv", index=False)
variable = long.groupby("Parameter").agg(rows=("value", "size"), measured_n=("value", "count"), minimum=("value", "min"), maximum=("value", "max")).reset_index()
for label, mask in [("minus_one_n", minus_one), ("negative_n", negative_invalid), ("zero_height_weight_n", zero_height_weight), ("documented_invalid_n", documented_invalid), ("duplicate_rows_n", duplicate)]:
    variable[label] = variable.Parameter.map(long.loc[mask, "Parameter"].value_counts()).fillna(0).astype(int)
measured_ids = long[long.value.notna()].groupby("Parameter").RecordID.nunique()
variable["admissions_measured_pct"] = 100 * variable.Parameter.map(measured_ids).fillna(0) / 4000
variable.to_csv(OUT / "variable_quality.csv", index=False)

def affected(mask):
    counts = long.loc[mask, "Parameter"].value_counts()
    return ", ".join(f"{name} ({count})" for name, count in counts.items()) or "None"

decision_rows = [
    ("Documented -1 missing code", affected(minus_one), int(minus_one.sum()), "Set numeric value to missing; retain value_raw", "PhysioNet defines -1 as missing or unknown."),
    ("Negative value other than -1", affected(negative_invalid), int(negative_invalid.sum()), "Set numeric value to missing; retain value_raw", "PhysioNet states that all valid values are non-negative."),
    ("pH outside documented 0-14 scale", affected(ph_invalid), int(ph_invalid.sum()), "Set numeric value to missing; retain value_raw", "PhysioNet documents arterial pH on a 0-14 scale."),
    ("Other documented-range or code violation", affected(gcs_invalid | fio2_invalid | gender_invalid | icu_invalid | vent_invalid), int((gcs_invalid | fio2_invalid | gender_invalid | icu_invalid | vent_invalid).sum()), "Set numeric value to missing if present", "PhysioNet documents the allowed scales or category codes."),
    ("Zero Height or Weight", affected(zero_height_weight), int(zero_height_weight.sum()), "Set numeric value to missing; retain value_raw", "A zero height or weight is not a plausible physiological measurement and is treated as a missing-value code."),
    ("Repeated RecordID-variable-time key", affected(duplicate), int(duplicate.sum()), "Retain for this checkpoint and report", "The source notes that measurements from different methods or sensors may share timestamps; the key alone cannot prove duplication."),
    ("Extreme but not documentation-invalid values", "Examples: Height 1.8-431.8 cm; Urine up to 11000 mL; HR up to 300 bpm", np.nan, "Retain and flag for sensitivity analysis", "PhysioNet warns that occasional outliers are expected; no unverified clinical cutoff is imposed."),
]
pd.DataFrame(decision_rows, columns=["issue", "affected_variables", "affected_rows", "decision", "rationale"]).to_csv(OUT / "anomaly_decisions.csv", index=False)

dynamic = long[~long.Parameter.isin(STATIC_NAMES + ["RecordID"])].sort_values(["RecordID", "Parameter", "time_hours"])
wide = build_modeling_table(long, source_ids)
wide = wide.merge(outcomes, on="RecordID", how="left", validate="one_to_one")
assert len(wide) == 4000 and not wide.RecordID.duplicated().any()
assert wide["In-hospital_death"].notna().all()
wide.to_csv(OUT / "set-a_wide.csv", index=False)

table = wide[["Age", "Gender", "ICUType", "Height", "Weight"]].copy()
table.Gender = table.Gender.map({0: "Female", 1: "Male"})
table.ICUType = table.ICUType.map({1: "Coronary care", 2: "Cardiac surgery", 3: "Medical ICU", 4: "Surgical ICU"})
table_rows = [("Admissions", "4,000", 4000, 0)]
for name in ["Age", "Height", "Weight"]:
    x = table[name]
    table_rows.append((name, f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]", int(x.notna().sum()), int(x.isna().sum())))
for name in ["Gender", "ICUType"]:
    valid = int(table[name].notna().sum())
    for category, n in table[name].value_counts().items():
        table_rows.append((f"{name}: {category}", f"{n} ({100*n/valid:.1f}%)", valid, 4000-valid))
pd.DataFrame(table_rows, columns=["variable", "summary", "valid_n", "missing_n"]).to_csv(OUT / "table1.csv", index=False)
outcome_rows = []
for name in ["SAPS-I", "SOFA", "Length_of_stay", "Survival"]:
    x = outcomes[name]
    outcome_rows.append((name, f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]", int(x.notna().sum()), int(x.isna().sum())))
pd.DataFrame(outcome_rows, columns=["outcome", "median_q1_q3", "valid_n", "missing_n"]).to_csv(OUT / "outcome_summary.csv", index=False)
outcomes["In-hospital_death"].value_counts().sort_index().rename("n").to_frame().assign(percent=lambda x: 100*x.n/4000).to_csv(OUT / "death_summary.csv")

valid_dynamic = dynamic[dynamic.value.notna() & dynamic.time_hours.between(0, 48)].copy()
valid_dynamic["bin_start"] = np.minimum(42, np.floor(valid_dynamic.time_hours / 6).astype(int) * 6)
parameters = sorted(dynamic.Parameter.unique())
presence = valid_dynamic.groupby(["RecordID", "Parameter", "bin_start"]).size().gt(0)
grid = pd.MultiIndex.from_product([source_ids, parameters, range(0, 48, 6)], names=["RecordID", "Parameter", "bin_start"])
presence = presence.reindex(grid, fill_value=False).rename("measured").reset_index()
rate = presence.groupby(["Parameter", "bin_start"]).measured.mean().reset_index(name="proportion_measured")
rate.to_csv(OUT / "temporal_missingness.csv", index=False)
dynamic_quality = variable[variable.Parameter.isin(parameters)].copy()
first_rate = rate[rate.bin_start.eq(0)].set_index("Parameter").proportion_measured
last_rate = rate[rate.bin_start.eq(42)].set_index("Parameter").proportion_measured
dynamic_quality["mean_measurements_per_admission"] = dynamic_quality.rows / len(source_ids)
dynamic_quality["measured_0_6h_pct"] = 100 * dynamic_quality.Parameter.map(first_rate)
dynamic_quality["measured_42_48h_pct"] = 100 * dynamic_quality.Parameter.map(last_rate)
dynamic_quality["change_first_to_last_pct_points"] = dynamic_quality.measured_42_48h_pct - dynamic_quality.measured_0_6h_pct
dynamic_quality[["Parameter", "admissions_measured_pct", "mean_measurements_per_admission", "measured_0_6h_pct", "measured_42_48h_pct", "change_first_to_last_pct_points"]].to_csv(OUT / "long_missingness_summary.csv", index=False)
selected = dynamic_quality.sort_values("admissions_measured_pct", ascending=False).Parameter.head(20).tolist()
plot = presence[presence.Parameter.isin(selected) & presence.RecordID.isin(sorted(source_ids)[:200])].copy()
plot["feature"] = plot.Parameter + " " + plot.bin_start.astype(str) + "h"
matrix = plot.pivot(index="RecordID", columns="feature", values="measured").astype(int)
fig, ax = plt.subplots(figsize=(16, 7))
sns.heatmap(matrix, cmap=["#f7f7f7", "#2166ac"], cbar=False, xticklabels=False, ax=ax)
ax.set(title="Measurement presence by six-hour interval, first 200 admissions", xlabel="Variable and interval", ylabel="Admission")
fig.tight_layout(); fig.savefig(FIG / "long_missingness_map.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(rate[rate.Parameter.isin(selected)].pivot(index="Parameter", columns="bin_start", values="proportion_measured"), vmin=0, vmax=1, cmap="viridis", ax=ax)
ax.set(title="Proportion of admissions measured by six-hour interval", xlabel="Hours after admission", ylabel="Variable")
fig.tight_layout(); fig.savefig(FIG / "variable_time_measurement_rate.png", dpi=160); plt.close(fig)

feature_columns = [c for c in wide if c != "RecordID" and not c.endswith("_count")]
missing = wide[feature_columns].isna().mean().sort_values(ascending=False)
missing.rename("missing_proportion").to_csv(OUT / "wide_missingness_summary.csv")
fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(wide[missing.head(60).index].isna().iloc[:400].astype(int), cmap=["#2166ac", "#f7f7f7"], cbar=False, xticklabels=False, ax=ax)
ax.set(title="Wide-table missingness, 60 features and first 400 admissions", xlabel="Patient-level feature", ylabel="Admission")
fig.tight_layout(); fig.savefig(FIG / "wide_missingness_map.png", dpi=160); plt.close(fig)

ever = valid_dynamic.groupby(["RecordID", "Parameter"]).size().gt(0)
ever_grid = pd.MultiIndex.from_product([source_ids, parameters], names=["RecordID", "Parameter"])
ever = ever.reindex(ever_grid, fill_value=False).rename("measured").reset_index()
ever["death"] = ever.RecordID.map(outcomes.set_index("RecordID")["In-hospital_death"])
informative = ever.groupby("Parameter").apply(lambda x: pd.Series({"measured_n": int(x.measured.sum()), "never_measured_n": int((~x.measured).sum()), "death_rate_measured": x.loc[x.measured, "death"].mean(), "death_rate_never_measured": x.loc[~x.measured, "death"].mean()}), include_groups=False).reset_index()
informative.to_csv(OUT / "missingness_by_outcome.csv", index=False)
print(f"Read {len(source_ids)} admissions and {len(long):,} measurements; wrote {len(wide):,} wide rows.")
