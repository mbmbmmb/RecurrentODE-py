"""For each continuous feature, compute post-3sigma-clip skew under
four transforms (raw, log, log1p, sqrt(|x|)) and rank them by |skew|.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'
CONTINUOUS = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
]


def post_clip_skew(x):
    s = pd.Series(x).dropna().astype(float)
    if len(s) == 0 or s.std(ddof=1) == 0:
        return np.nan
    m, sd = s.mean(), s.std(ddof=1)
    s = s.clip(m - 3 * sd, m + 3 * sd)
    return float(stats.skew(s, bias=False))


def main():
    df = pd.read_csv(os.path.join(OUT, 'cohort_clean.csv'))
    rows = []
    for col in CONTINUOUS:
        s = df[col].dropna().astype(float).values
        skews = {
            'raw':       post_clip_skew(s),
            'log(x)':    post_clip_skew(np.log(s))   if s.min() > 0 else np.nan,
            'log1p(x)':  post_clip_skew(np.log1p(s)) if s.min() >= 0 else np.nan,
            'sqrt(|x|)': post_clip_skew(np.sqrt(np.abs(s))),
        }
        absk = {k: (abs(v) if not np.isnan(v) else np.inf) for k, v in skews.items()}
        best = min(absk, key=absk.get)
        rows.append({
            'feature': col,
            'raw':       f'{skews["raw"]:+.3f}',
            'log(x)':    '   N/A' if np.isnan(skews['log(x)']) else f'{skews["log(x)"]:+.3f}',
            'log1p(x)':  '   N/A' if np.isnan(skews['log1p(x)']) else f'{skews["log1p(x)"]:+.3f}',
            'sqrt(|x|)': f'{skews["sqrt(|x|)"]:+.3f}',
            'best':      best,
            '|skew_best|': f'{absk[best]:.3f}',
        })
    rep = pd.DataFrame(rows).sort_values(
        '|skew_best|', key=lambda s: s.astype(float))
    print(rep.to_string(index=False))
    print('\n=== Best-transform tally ===')
    print(rep['best'].value_counts().to_string())
    rep.to_csv(os.path.join(OUT, 'transform_ranking.csv'), index=False)


if __name__ == '__main__':
    main()
