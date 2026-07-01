"""Investigate the 339 admissions in input.csv that have no label.csv entry.

Compares the unlabeled cohort to the labeled cohort across:
- admission type / discharge location / hospital_expire_flag
- demographics (gender, age at admission)
- length of stay
- whether the same SUBJECT_ID has *other* labeled admissions

If a patient died in-hospital (HOSPITAL_EXPIRE_FLAG=1 or non-null DEATHTIME),
the survival outcome is fully recoverable from raw-data even if label.csv
omitted it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/MIMIC Data'
SRC = os.path.join(ROOT, 'processed-data')
RAW = os.path.join(ROOT, 'raw-data')
OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'


def main():
    os.makedirs(OUT, exist_ok=True)
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None,
                       names=['HADM_ID']).astype({'HADM_ID': 'int64'})
    inp = pd.read_csv(os.path.join(SRC, 'input.csv'), header=None)
    label = pd.read_csv(os.path.join(SRC, 'label.csv')).astype({'HADM_ID': 'int64'})
    adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])
    pat = pd.read_csv(os.path.join(RAW, 'PATIENTS.csv'),
                      parse_dates=['DOB', 'DOD'])

    labeled_ids = set(label['HADM_ID'])
    unlabeled_ids = set(aids['HADM_ID']) - labeled_ids
    print(f'labeled : {len(labeled_ids)}')
    print(f'unlabeled: {len(unlabeled_ids)}')

    aids = aids.assign(has_label=aids['HADM_ID'].isin(labeled_ids))
    aids = aids.assign(age=inp.iloc[:, 0].values,  # V1 = age (years)
                       AIDS=inp.iloc[:, 22].values,
                       HEM=inp.iloc[:, 23].values,
                       METS=inp.iloc[:, 24].values,
                       Adm_type=inp.iloc[:, 25].values)

    j = aids.merge(adm, on='HADM_ID', how='left')
    j = j.merge(pat[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']],
                on='SUBJECT_ID', how='left')
    j['LOS_days'] = (j['DISCHTIME'] - j['ADMITTIME']).dt.total_seconds() / 86400
    j['died_in_hospital'] = j['HOSPITAL_EXPIRE_FLAG'] == 1
    j['has_DEATHTIME'] = j['DEATHTIME'].notna()
    j['has_DOD'] = j['DOD'].notna()

    grp = j.groupby('has_label')

    def pct(s):
        return f'{100 * s.mean():.1f}%'

    print('\n=== Admission characteristics by label presence ===')
    rows = []
    for has, sub in grp:
        rows.append({
            'has_label': has,
            'n': len(sub),
            'mean_age': float(sub['age'].mean()),
            'median_age': float(sub['age'].median()),
            'pct_female': pct(sub['GENDER'] == 'F'),
            'mean_LOS_days': float(sub['LOS_days'].mean()),
            'median_LOS_days': float(sub['LOS_days'].median()),
            'pct_died_in_hospital': pct(sub['died_in_hospital']),
            'pct_has_DEATHTIME': pct(sub['has_DEATHTIME']),
            'pct_has_DOD': pct(sub['has_DOD']),
            'pct_AIDS': pct(sub['AIDS'] == 1),
            'pct_HEM': pct(sub['HEM'] == 1),
            'pct_METS': pct(sub['METS'] == 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n=== Categorical breakdown of UNLABELED admissions ===')
    unl = j[~j['has_label']]
    for col in ['ADMISSION_TYPE', 'DISCHARGE_LOCATION',
                'HOSPITAL_EXPIRE_FLAG']:
        print(f'\n{col}:')
        ct = (unl[col].value_counts(dropna=False)
              .rename_axis(col).reset_index(name='unlabeled'))
        # Compare to baseline rate in labeled
        lab_rate = (j[j['has_label']][col]
                    .value_counts(normalize=True, dropna=False))
        ct['unlabeled_pct'] = (ct['unlabeled'] / len(unl)).map(
            lambda x: f'{100 * x:.1f}%')
        ct['labeled_pct'] = ct[col].map(
            lambda v: f'{100 * lab_rate.get(v, 0):.1f}%')
        print(ct.to_string(index=False))

    print('\n=== Same patient multiple admissions? ===')
    n_subj_unl = unl['SUBJECT_ID'].nunique()
    print(f'339 unlabeled admissions span {n_subj_unl} unique SUBJECT_IDs')
    counts = j.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    unl_counts = unl.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    print('admissions per subject (unlabeled-cohort subjects, '
          'considering ALL their admissions in input.csv):')
    print(unl_counts.value_counts().sort_index().to_string())

    # For unlabeled-cohort subjects, do they ALSO have labeled admissions?
    overlap = j[j['SUBJECT_ID'].isin(unl['SUBJECT_ID'].unique())]
    has_other_labeled = (
        overlap[overlap['has_label']]['SUBJECT_ID'].nunique()
    )
    print(f'\nof {n_subj_unl} unlabeled-cohort subjects, '
          f'{has_other_labeled} have at least one OTHER labeled admission')

    print('\n=== Recoverability of survival outcome from raw data ===')
    # An outcome is "recoverable" if we know the time-to-death OR can right-
    # censor at discharge: we know DISCHTIME for everyone and DOD when died.
    n_recoverable = int(((unl['has_DEATHTIME'] | unl['has_DOD']) |
                         unl['DISCHTIME'].notna()).sum())
    n_with_dod = int(unl['has_DOD'].sum())
    n_died_inhosp = int(unl['died_in_hospital'].sum())
    print(f'{n_recoverable}/{len(unl)} unlabeled admissions could in '
          f'principle have label imputed from raw-data')
    print(f'  - died in hospital (HOSPITAL_EXPIRE_FLAG=1): {n_died_inhosp}')
    print(f'  - have DOD (date of death) on file: {n_with_dod}')
    print(f'  - none died and no DOD recorded -> right-censored at DISCHTIME')

    out_path = os.path.join(OUT, 'unlabeled_diagnostic.csv')
    cols = ['HADM_ID', 'SUBJECT_ID', 'GENDER', 'age', 'ADMISSION_TYPE',
            'DISCHARGE_LOCATION', 'HOSPITAL_EXPIRE_FLAG', 'DEATHTIME',
            'DOD', 'EXPIRE_FLAG', 'LOS_days', 'AIDS', 'HEM', 'METS',
            'Adm_type']
    unl[cols].to_csv(out_path, index=False)
    print(f'\nwrote {out_path}: {unl.shape}')


if __name__ == '__main__':
    main()
