from pathlib import Path
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
DATA = Path(os.environ.get('BIOSTAT_DATA', ROOT / 'data'))
OUT = ROOT / 'output'
FIG = OUT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')

files = sorted((DATA / 'set-a').glob('*.txt'))
assert len(files) == 4000

def parse_time(value):
    h, m = str(value).split(':')
    return int(h) + int(m) / 60

def read_one(path):
    rows = []
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == ['Time', 'Parameter', 'Value']
        for row in reader:
            raw = row['Value']
            rows.append([int(path.stem), row['Time'], parse_time(row['Time']), row['Parameter'], raw, np.nan if raw == '-1' else float(raw)])
    return rows

rows = []
with ThreadPoolExecutor(max_workers=16) as pool:
    for block in pool.map(read_one, files):
        rows.extend(block)
long = pd.DataFrame(rows, columns=['RecordID', 'Time', 'time_hours', 'Parameter', 'value_raw', 'value'])
long.to_csv(OUT / 'set-a_long.csv', index=False)
outcomes = pd.read_csv(DATA / 'Outcomes-a.txt')
outcomes['RecordID'] = outcomes['RecordID'].astype(int)
assert long['RecordID'].nunique() == 4000
assert set(long['RecordID']) == set(outcomes['RecordID'])

static_names = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
static = (long[long.Parameter.isin(static_names)].sort_values(['RecordID', 'Parameter', 'time_hours'])
          .drop_duplicates(['RecordID', 'Parameter']).pivot(index='RecordID', columns='Parameter', values='value').reset_index())
dynamic = long[~long.Parameter.isin(static_names)]
summary = dynamic.groupby(['RecordID', 'Parameter'])['value'].agg(['count', 'first', 'last', 'min', 'max', 'mean']).reset_index()
wide_dynamic = summary.pivot(index='RecordID', columns='Parameter')
wide_dynamic.columns = [f'{variable}_{stat}' for stat, variable in wide_dynamic.columns]
wide = static.merge(wide_dynamic.reset_index(), on='RecordID', how='outer').merge(outcomes, on='RecordID', how='left')
wide.to_csv(OUT / 'set-a_wide.csv', index=False)

table1 = wide[['Age', 'Gender', 'ICUType', 'Height', 'Weight', 'In-hospital_death']].copy()
table1['Gender'] = table1.Gender.map({0: 'Female', 1: 'Male'})
table1['ICUType'] = table1.ICUType.map({1: 'Coronary care unit', 2: 'Cardiac surgical ICU', 3: 'Medical ICU', 4: 'Surgical ICU'})
rows = []
for variable in ['Age', 'Height', 'Weight']:
    x = pd.to_numeric(table1[variable], errors='coerce')
    rows.append({'variable': variable, 'summary': f'{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]', 'missing_n': int(x.isna().sum())})
for variable in ['Gender', 'ICUType', 'In-hospital_death']:
    for level, n in table1[variable].value_counts(dropna=False).items():
        rows.append({'variable': f'{variable}: {level if pd.notna(level) else "Missing"}', 'summary': f'{n} ({100*n/len(table1):.1f}%)', 'missing_n': int(table1[variable].isna().sum())})
pd.DataFrame(rows).to_csv(OUT / 'table1.csv', index=False)
outcomes.describe().T.to_csv(OUT / 'outcome_summary.csv')
outcomes['In-hospital_death'].value_counts(dropna=False).rename('n').to_frame().assign(percent=lambda x: 100*x.n/len(outcomes)).to_csv(OUT / 'death_summary.csv')

long['time_bin'] = pd.cut(long.time_hours, bins=[0, 6, 12, 18, 24, 30, 36, 42, 48], right=False, include_lowest=True)
presence = long.assign(measured=long.value.notna()).groupby(['RecordID', 'Parameter', 'time_bin'], observed=False).measured.any().reset_index()
heat = presence.pivot_table(index='RecordID', columns=['Parameter', 'time_bin'], values='measured', fill_value=False)
heat = heat.astype(int)
fig, ax = plt.subplots(figsize=(16, 7)); sns.heatmap(heat.iloc[:200], cmap=['white', '#2166ac'], cbar=False, ax=ax); ax.set(title='Long-table measurement presence by six-hour interval (first 200 admissions)', xlabel='Variable and time interval', ylabel='Admission'); fig.tight_layout(); fig.savefig(FIG / 'long_missingness_map.png', dpi=160); plt.close(fig)
rate = presence.groupby(['Parameter', 'time_bin'], observed=False).measured.mean().reset_index()
top = long.groupby('Parameter').RecordID.nunique().nlargest(25).index
fig, ax = plt.subplots(figsize=(14, 7)); sns.heatmap(rate[rate.Parameter.isin(top)].pivot(index='Parameter', columns='time_bin', values='measured'), vmin=0, vmax=1, cmap='viridis', ax=ax); ax.set(title='Proportion of admissions measured by variable and six-hour interval', xlabel='Time after admission', ylabel='Variable'); fig.tight_layout(); fig.savefig(FIG / 'variable_time_measurement_rate.png', dpi=160); plt.close(fig)
wide.isna().mean().sort_values(ascending=False).rename('missing_proportion').to_csv(OUT / 'wide_missingness_summary.csv')
fig, ax = plt.subplots(figsize=(16, 7)); sns.heatmap(wide.drop(columns='RecordID').isna().iloc[:, :80], cmap=['#2166ac', '#f7f7f7'], cbar=False, ax=ax); ax.set(title='Wide-table missingness map (first 80 features)', xlabel='Feature', ylabel='Admission'); fig.tight_layout(); fig.savefig(FIG / 'wide_missingness_map.png', dpi=160); plt.close(fig)
print({'files': len(files), 'long_rows': len(long), 'records': long.RecordID.nunique(), 'variables': long.Parameter.nunique(), 'time_min': long.time_hours.min(), 'time_max': long.time_hours.max(), 'death_n': int(outcomes['In-hospital_death'].sum())})
