"""Merge MIMIC-III processed-data CSVs into a single labeled dataframe.

- ``input.csv``      : 35,643 rows x 26 columns of summary statistics (no header).
- ``valid_aids.csv`` : 35,643 rows x 1 column = HADM_ID (no header). Aligned by
  row with ``input.csv``.
- ``label.csv``      : 35,304 rows; columns HADM_ID, censor, time.

Outputs (under ``mimic_analysis/merged_data/``):
- ``input_named.csv`` : input.csv + HADM_ID with V1..V26 mapped to names.
- ``merged.csv``      : input_named left-joined with label.csv on HADM_ID.
"""
from __future__ import annotations

import os

import pandas as pd

SRC = '/Users/bomeng/Desktop/research/review/jmlr/code/MIMIC Data/processed-data'
OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'

# V1..V26 names from the PDF (Table on pages 1-2).
V_NAMES = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
    'AIDS', 'HEM', 'METS', 'Adm_type',
]
assert len(V_NAMES) == 26


def main():
    os.makedirs(OUT, exist_ok=True)

    inp = pd.read_csv(os.path.join(SRC, 'input.csv'), header=None)
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None)
    lab = pd.read_csv(os.path.join(SRC, 'label.csv'))

    print(f'input.csv      : {inp.shape}')
    print(f'valid_aids.csv : {aids.shape}')
    print(f'label.csv      : {lab.shape}')

    if inp.shape[0] != aids.shape[0]:
        raise ValueError(
            f'row mismatch: input has {inp.shape[0]}, '
            f'valid_aids has {aids.shape[0]}',
        )
    if inp.shape[1] != len(V_NAMES):
        raise ValueError(
            f'column mismatch: input has {inp.shape[1]}, '
            f'expected {len(V_NAMES)}',
        )

    inp.columns = V_NAMES
    aids.columns = ['HADM_ID']
    aids['HADM_ID'] = aids['HADM_ID'].astype('int64')

    # Step 1: row-wise concat (input + HADM_ID).
    input_named = pd.concat([aids, inp], axis=1)
    out_named = os.path.join(OUT, 'input_named.csv')
    input_named.to_csv(out_named, index=False)
    print(f'wrote {out_named}: {input_named.shape}')

    # Step 2: attach label.csv on HADM_ID. Left join keeps every input row;
    # rows without a label end up with NaN in censor/time.
    lab['HADM_ID'] = lab['HADM_ID'].astype('int64')
    merged = input_named.merge(lab, on='HADM_ID', how='left')

    out_merged = os.path.join(OUT, 'merged.csv')
    merged.to_csv(out_merged, index=False)
    print(f'wrote {out_merged}: {merged.shape}')

    n_with_label = merged['time'].notna().sum()
    n_event = int((merged['censor'] == 1).sum())
    n_censored = int((merged['censor'] == 0).sum())
    print(
        f'label coverage : {n_with_label}/{len(merged)} '
        f'({100 * n_with_label / len(merged):.1f}%) have label; '
        f'events={n_event}, censored={n_censored}'
    )


if __name__ == '__main__':
    main()
