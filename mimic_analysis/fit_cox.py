"""Random-effect (gamma frailty) Cox PH on the readmission cohort.

Uses the local ``recurrent_ode`` package directly (not lifelines): the
spline-baseline Cox-with-frailty estimator from ``recurrent_ode.fit``
with ``ci=True`` to get the closed-form sandwich SE for ``beta``.

Three scenarios:
  1. RAW   : time = days since first admit (as stored in the npz).
  2. LOG10 : same rows, but log10(1 + time) replacing time
             (keeps the transform non-negative so the spline knots
             on [0, max] stay valid).
  3. TRIM  : drop rows whose `time` is outside the middle 99% of the
             time distribution, then refit on raw time.

Outputs:
  - merged_data/cox_coef_table.csv  (long format: feature, scenario, ...)
  - merged_data/cox_coef_wide.csv   (wide: feature x scenario coef/se/p)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ  = os.path.join(ROOT, 'merged_data/cohort_long_scaled.npz')
OUT  = os.path.join(ROOT, 'merged_data')

# Make recurrent_ode importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recurrent_ode import fit  # noqa: E402

# Adm_type one-hot reference: medical (0). Two dummies retained.
ADM_DUMMIES = ['Adm_type_scheduled', 'Adm_type_unscheduled']


def load_long():
    npz = np.load(NPZ, allow_pickle=True)
    cols = [c.decode() if isinstance(c, bytes) else str(c)
            for c in npz['x_cols']]
    df = pd.DataFrame(npz['x'], columns=cols)
    df['id']    = npz['id'].astype(int)
    df['time']  = npz['time'].astype(float)
    df['delta'] = npz['delta'].astype(int)
    return df, cols


def encode_admtype(df):
    """One-hot Adm_type {0,1,2} into two dummies; drop the original."""
    out = df.copy()
    out['Adm_type_scheduled']   = (out['Adm_type'] == 1).astype(float)
    out['Adm_type_unscheduled'] = (out['Adm_type'] == 2).astype(float)
    return out.drop(columns=['Adm_type'])


def remap_ids(df):
    """Remap subject ids to 1..N consecutive integers (required by the
    recurrent_ode sandwich-variance code)."""
    out = df.copy()
    uniq = np.sort(out['id'].unique())
    mapping = {old: i + 1 for i, old in enumerate(uniq)}
    out['id'] = out['id'].map(mapping).astype(int)
    return out, len(uniq)


def fit_one(df, cols, label, time_xform=None):
    """Fit random-effect Cox; return a per-feature summary frame."""
    work = encode_admtype(df).copy()
    if time_xform is not None:
        work['time'] = time_xform(work['time'].values)

    feats = [c for c in cols if c != 'Adm_type'] + ADM_DUMMIES
    work, N = remap_ids(work)
    work = work.sort_values(['id', 'time']).reset_index(drop=True)

    x      = work[feats].to_numpy(dtype=float)
    time_  = work['time'].to_numpy(dtype=float)
    delta  = work['delta'].to_numpy(dtype=int)
    id_vec = work['id'].to_numpy(dtype=int)

    n_evt = int((delta == 1).sum())
    n_cen = int((delta == 0).sum())
    print(f'\n=== Scenario: {label}  '
          f'(rows={len(work)}, patients={N}, '
          f'events={n_evt}, censors={n_cen}, '
          f'time range=[{time_.min():.4f}, {time_.max():.4f}]) ===')

    data = {'x': x, 'time': time_, 'delta': delta, 'id': id_vec}
    est = fit(data, model='cox', random_effect=True, ci=True)

    coef = est.beta.ravel()
    se   = est.se.ravel()
    z    = coef / se
    pval = 2.0 * norm.sf(np.abs(z))
    lo95 = coef - 1.96 * se
    hi95 = coef + 1.96 * se

    s = pd.DataFrame({
        'feature': feats,
        'scenario': label,
        'coef': coef, 'se': se, 'z': z, 'p': pval,
        'lo95': lo95, 'hi95': hi95,
    })
    print(f'  runtime: {est.runtime:.1f}s')
    print(s.to_string(index=False, float_format=lambda v: f'{v:8.4f}'))
    return s, est


def main():
    df, cols = load_long()
    print(f'cohort_long_scaled : rows={len(df)}  '
          f'patients={df["id"].nunique()}  '
          f'events={int((df["delta"] == 1).sum())}  '
          f'censors={int((df["delta"] == 0).sum())}')

    # Scenario 1: raw -----------------------------------------------------
    s_raw, est_raw = fit_one(df, cols, 'raw')

    # Scenario 2: log10(1 + t) -------------------------------------------
    # The spline baseline-hazard knots are placed on [0, max(time)], so the
    # transformed time has to stay non-negative. log10(1 + t) preserves the
    # heavy-right-tail compression of log10(t) while pinning t = 0 -> 0.
    s_log, est_log = fit_one(df, cols, 'log10',
                       time_xform=lambda t: np.log10(1.0 + t))

    # Save baseline-hazard splines for downstream visualization.
    for est, label in [(est_raw, 'raw'), (est_log, 'log10')]:
        np.savez_compressed(
            os.path.join(OUT, f'cox_spline_{label}.npz'),
            theta=est.spline['coefs'],
            knots=est.spline['knots'],
            k=est.spline['k'],
            fish=est.raw.get('fish', np.array([])),
            beta=est.beta,
            se=est.se,
        )

    # Scenario 3: trim outliers in `time` (q01 -- q99) --------------------
    q01, q99 = df['time'].quantile([0.01, 0.99]).values
    print(f'\ntrim bounds (middle 99% of `time`): '
          f'[{q01:.3f}, {q99:.3f}] days')
    df_trim = df[(df['time'] >= q01) & (df['time'] <= q99)].copy()
    print(f'rows kept after trim: {len(df_trim)} / {len(df)}  '
          f'(events kept: {int((df_trim["delta"] == 1).sum())}, '
          f'patients kept: {df_trim["id"].nunique()})')
    # Drop patients who lost their censoring row in the trim.
    has_cen = df_trim.groupby('id')['delta'].apply(lambda s: (s == 0).any())
    keep_ids = has_cen[has_cen].index
    df_trim = df_trim[df_trim['id'].isin(keep_ids)].copy()
    print(f'rows after enforcing >=1 censoring per patient: '
          f'{len(df_trim)} / {len(df)}')
    s_trim, _ = fit_one(df_trim, cols, 'trim')

    # ------- Combine & persist -------------------------------------------
    long = pd.concat([s_raw, s_log, s_trim], ignore_index=True)
    long_path = os.path.join(OUT, 'cox_coef_table.csv')
    long.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long.shape}')

    pivot = long.pivot(index='feature', columns='scenario',
                       values=['coef', 'se', 'p'])
    pivot.columns = [f'{m}_{s}' for m, s in pivot.columns]
    pivot = pivot.reindex(columns=[
        'coef_raw',  'se_raw',  'p_raw',
        'coef_log10','se_log10','p_log10',
        'coef_trim', 'se_trim', 'p_trim'])
    feat_order = s_raw['feature'].tolist()
    pivot = pivot.reindex(feat_order)
    wide_path = os.path.join(OUT, 'cox_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')

    # ------- Significance summary ----------------------------------------
    # Wald p-values use the closed-form sandwich SE (inference_beta), so
    # they're the per-feature exact significance under the model -- no
    # multiple-testing correction is applied.
    print('\n=== Significant features (Wald p < 0.05, sandwich SE) ===')
    for sc in ['raw', 'log10', 'trim']:
        sig = pivot[pivot[f'p_{sc}'] < 0.05].index.tolist()
        print(f'  {sc:5s}: {len(sig)} sig  -> {sig}')


if __name__ == '__main__':
    main()
