"""Verify two claims about the cohort:

(A) The HADM_ID in input.csv is the patient's FIRST hospital admission
    (the PDF says "first ICU admission" — we check the hospital level too).
(B) Examine the rows with DISCHTIME < ADMITTIME (negative LOS): are these
    all in-hospital deaths on the first admission? Did they belong to the
    cohort?
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/MIMIC Data'
SRC = os.path.join(ROOT, 'processed-data')
RAW = os.path.join(ROOT, 'raw-data')


def main():
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None,
                       names=['HADM_ID']).astype({'HADM_ID': 'int64'})
    adm = pd.read_csv(os.path.join(RAW, 'ADMISSIONS.csv'),
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                               'DISCHTIME', 'DEATHTIME',
                               'HOSPITAL_EXPIRE_FLAG'])
    cohort = aids.merge(adm, on='HADM_ID', how='left')

    # ---- (A) Is the indexed admission the patient's first hospitalization?
    print('=== (A) Is input.csv HADM_ID = patient first admission? ===')
    earliest = (adm.sort_values('ADMITTIME')
                  .drop_duplicates('SUBJECT_ID', keep='first')
                  [['SUBJECT_ID', 'HADM_ID']]
                  .rename(columns={'HADM_ID': 'first_HADM_ID'}))
    chk = cohort.merge(earliest, on='SUBJECT_ID', how='left')
    is_first = (chk['HADM_ID'] == chk['first_HADM_ID']).sum()
    n = len(chk)
    print(f'cohort size                     : {n}')
    print(f'indexed = first hospital admit  : {is_first} '
          f'({100 * is_first / n:.2f}%)')
    not_first = chk[chk['HADM_ID'] != chk['first_HADM_ID']]
    print(f'indexed != first hospital admit : {len(not_first)} '
          f'({100 * len(not_first) / n:.2f}%)')
    if len(not_first):
        # gap between actual first and indexed admission
        ft = adm.set_index('HADM_ID')['ADMITTIME']
        gap = (not_first['ADMITTIME'].values
               - ft.loc[not_first['first_HADM_ID']].values)
        gap_days = pd.Series(gap).dt.total_seconds() / 86400
        print(f'  median gap (days, indexed - first): '
              f'{gap_days.median():.1f}')
        print(f'  mean gap   (days, indexed - first): '
              f'{gap_days.mean():.1f}')
        print('Likely reason: indexed admission is the first ICU admission;')
        print('the patient may have had an earlier non-ICU hospital stay.')

    # ---- (B) DISCHTIME < ADMITTIME records ----
    print('\n=== (B) Records with DISCHTIME < ADMITTIME ===')
    bad = adm[adm['DISCHTIME'] < adm['ADMITTIME']].copy()
    print(f'count in ADMISSIONS.csv (whole table) : {len(bad)}')
    bad['LOS_hr'] = (bad['DISCHTIME'] - bad['ADMITTIME']).dt.total_seconds() / 3600
    print(f'  LOS (hr): min {bad["LOS_hr"].min():.2f}  '
          f'max {bad["LOS_hr"].max():.2f}  '
          f'median {bad["LOS_hr"].median():.2f}')
    print(f'  HOSPITAL_EXPIRE_FLAG=1 : '
          f'{int((bad["HOSPITAL_EXPIRE_FLAG"]==1).sum())} / {len(bad)} '
          f'({100*(bad["HOSPITAL_EXPIRE_FLAG"]==1).mean():.1f}%)')
    print(f'  has DEATHTIME          : '
          f'{int(bad["DEATHTIME"].notna().sum())} / {len(bad)}')

    in_cohort = bad[bad['HADM_ID'].isin(aids['HADM_ID'])]
    print(f'\nIn cohort (input.csv HADM_IDs): {len(in_cohort)}')
    if len(in_cohort):
        print(in_cohort[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                         'DISCHTIME', 'DEATHTIME',
                         'HOSPITAL_EXPIRE_FLAG', 'LOS_hr']].to_string(index=False))

    # And patients-with-negative-FU at the patient (recurrent) level:
    print('\n=== Patients with negative follow-up (last - first < 0) ===')
    g = (adm[adm['SUBJECT_ID'].isin(cohort['SUBJECT_ID'])]
         .groupby('SUBJECT_ID'))
    first_a = g['ADMITTIME'].min()
    last_d = g['DISCHTIME'].max()
    last_death = g['DEATHTIME'].max()
    last_t = last_death.fillna(last_d)
    fu = (last_t - first_a).dt.total_seconds() / 86400
    neg = fu[fu < 0]
    print(f'patients with negative follow-up: {len(neg)}')
    if len(neg):
        print(neg.describe().to_string())
        # Did they die in hospital on the first admission?
        first_hadm = (adm.sort_values('ADMITTIME')
                        .drop_duplicates('SUBJECT_ID', keep='first')
                        .set_index('SUBJECT_ID'))
        died_first = first_hadm.loc[neg.index, 'HOSPITAL_EXPIRE_FLAG'].sum()
        print(f'  of these, died in hospital on first admission: '
              f'{int(died_first)} / {len(neg)}')


if __name__ == '__main__':
    main()
