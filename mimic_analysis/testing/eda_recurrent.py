"""Initial EDA for the recurrent-event readmission setup.

For each patient (SUBJECT_ID):
  - First admission ADMITTIME -> time 0
  - Subsequent admissions -> recurrent events at (ADMITTIME - first_ADMITTIME)
  - Last DISCHTIME (or DEATHTIME) -> censoring time (informative if death)
  - X = the V1..V26 medical-record summary statistics from input.csv

Reports:
  1. Per-column missingness in V1..V26
  2. Pairwise Pearson correlation among V1..V26 (lists |r| >= 0.6 pairs and
     saves the full 26x26 matrix to merged_data/corr_matrix.csv)
  3. Per-patient distribution of admission / event counts and follow-up time
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/MIMIC Data'
SRC = os.path.join(ROOT, 'processed-data')
RAW = os.path.join(ROOT, 'raw-data')
OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'

V_NAMES = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
    'AIDS', 'HEM', 'METS', 'Adm_type',
]


def _col_missing(s: pd.Series) -> dict:
    n = len(s)
    n_nan = int(s.isna().sum())
    return {
        'n': n,
        'n_nan': n_nan,
        'pct_nan': round(100 * n_nan / n, 3),
        'n_zero': int((s == 0).sum()),
        'n_negative': int((s < 0).sum()),
        'min': float(s.min(skipna=True)),
        'max': float(s.max(skipna=True)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- Load -----------------------------------------------------------
    inp = pd.read_csv(os.path.join(SRC, 'input.csv'), header=None,
                      names=V_NAMES)
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None,
                       names=['HADM_ID']).astype({'HADM_ID': 'int64'})
    adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                               'DISCHTIME', 'DEATHTIME',
                               'HOSPITAL_EXPIRE_FLAG'])

    df = pd.concat([aids, inp], axis=1)
    df = df.merge(adm, on='HADM_ID', how='left')
    print(f'admissions in input.csv: {len(df)}')
    print(f'unique patients (SUBJECT_ID): {df["SUBJECT_ID"].nunique()}')

    # ---- 1. Missingness ------------------------------------------------
    print('\n=== 1. Missingness in V1..V26 ===')
    miss = pd.DataFrame({c: _col_missing(df[c]) for c in V_NAMES}).T
    miss = miss[['n', 'n_nan', 'pct_nan', 'n_zero', 'n_negative',
                 'min', 'max']]
    miss = miss.sort_values('pct_nan', ascending=False)
    print(miss.to_string())
    miss.to_csv(os.path.join(OUT, 'missingness.csv'))
    print(f'(saved to {os.path.join(OUT, "missingness.csv")})')

    # ---- 2. Correlation -----------------------------------------------
    print('\n=== 2. Correlation (Pearson) across V1..V26 ===')
    cor = df[V_NAMES].corr()
    cor.to_csv(os.path.join(OUT, 'corr_matrix.csv'))
    print(f'(full 26x26 matrix -> {os.path.join(OUT, "corr_matrix.csv")})')

    pairs = []
    for i, a in enumerate(V_NAMES):
        for b in V_NAMES[i + 1:]:
            r = cor.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.6:
                pairs.append((abs(r), a, b, r))
    pairs.sort(reverse=True)
    print(f'\nPairs with |r| >= 0.6 ({len(pairs)} found):')
    for _, a, b, r in pairs:
        flag = '  ***' if abs(r) >= 0.8 else ''
        print(f'  {a:20s}  {b:20s}  r = {r:+.3f}{flag}')

    # ---- 3. Per-patient summary ---------------------------------------
    # IMPORTANT: input.csv has one row per patient (35,643 admissions =
    # 35,643 unique SUBJECT_IDs). To build the recurrent-event view we
    # have to pull ALL admissions per subject from ADMISSIONS.csv, not
    # just the indexed admission that has X covariates.
    print('\n=== 3. Per-patient recurrent-event summary ===')
    cohort = df['SUBJECT_ID'].dropna().unique()
    all_adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                          parse_dates=['ADMITTIME', 'DISCHTIME',
                                       'DEATHTIME'],
                          usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                                   'DISCHTIME', 'DEATHTIME',
                                   'HOSPITAL_EXPIRE_FLAG'])
    all_adm = all_adm[all_adm['SUBJECT_ID'].isin(cohort)].sort_values(
        ['SUBJECT_ID', 'ADMITTIME'])
    print(f'cohort patients              : {len(cohort)}')
    print(f'admissions for these patients: {len(all_adm)} '
          f'(from ADMISSIONS.csv, all visits)')

    g = all_adm.groupby('SUBJECT_ID', sort=False)
    n_adm = g.size()
    n_evt = n_adm - 1                       # events = readmissions
    first_admit = g['ADMITTIME'].min()
    last_dis = g['DISCHTIME'].max()
    last_death = g['DEATHTIME'].max()
    censor_dt = last_death.fillna(last_dis)
    fu_days = (censor_dt - first_admit).dt.total_seconds() / 86400
    last_died = g['HOSPITAL_EXPIRE_FLAG'].apply(lambda s: int(s.iloc[-1]))

    pp = pd.DataFrame({
        'n_admissions': n_adm,
        'n_events': n_evt,
        'follow_up_days': fu_days,
        'died_at_end': last_died,
    })

    n_pts = len(pp)
    print(f'patients               : {n_pts}')
    print(f'total admissions       : {int(n_adm.sum())}')
    print(f'total events (readmits): {int(n_evt.sum())}')
    print(f'mean events / patient  : {n_evt.mean():.3f}')
    print(f'median events / patient: {n_evt.median():.0f}')
    n_zero = int((pp['n_events'] == 0).sum())
    print(f'patients with 0 events : {n_zero} '
          f'({100 * n_zero / n_pts:.1f}%)')

    print('\nAdmissions-per-patient distribution:')
    vc = pp['n_admissions'].value_counts().sort_index()
    head = vc.head(10)
    print(head.to_string())
    if len(vc) > 10:
        print(f'  ... (max = {pp["n_admissions"].max()})')

    print('\nDescribe (n_events, follow_up_days):')
    print(pp[['n_events', 'follow_up_days']].describe().to_string())

    n_died = int(pp['died_at_end'].sum())
    print(f'\npatients whose final admission ended in death: '
          f'{n_died} ({100 * n_died / n_pts:.1f}%)')
    print('  -> contribute informative (death-censored) follow-ups')

    pp_out = os.path.join(OUT, 'per_patient_summary.csv')
    pp.reset_index().to_csv(pp_out, index=False)
    print(f'\nwrote {pp_out}: {pp.shape}')


if __name__ == '__main__':
    main()
