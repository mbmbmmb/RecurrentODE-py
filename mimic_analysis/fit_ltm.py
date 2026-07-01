"""Random-effect LTM fit on the readmission cohort.

The LTM model fixes ``beta_1 = 1`` for identifiability, so we put the
strongest Cox feature (``bicarbonate_mean``) first as anchor. All other
LTM coefficients are then on the same scale: they are *relative effects*
vs. bicarbonate_mean.

Knots: ``K1`` (uniform on [0, max(time)] for both alpha and the baseline
hazard); higher-order quantile-based knots (K2..K4) hit numerical
overflow in the iterative MLE on this cohort.

Output:
  - merged_data/ltm_coef_table.csv  (raw, log10 scenarios)
"""
from __future__ import annotations

import os
import sys
import time as _time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ  = os.path.join(ROOT, 'merged_data/cohort_long_scaled.npz')
OUT  = os.path.join(ROOT, 'merged_data')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recurrent_ode import fit  # noqa: E402

ADM_DUMMIES = ['Adm_type_scheduled', 'Adm_type_unscheduled']
ANCHOR = 'bicarbonate_mean'


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
    out = df.copy()
    out['Adm_type_scheduled']   = (out['Adm_type'] == 1).astype(float)
    out['Adm_type_unscheduled'] = (out['Adm_type'] == 2).astype(float)
    return out.drop(columns=['Adm_type'])


def remap_ids(df):
    out = df.copy()
    uniq = np.sort(out['id'].unique())
    mapping = {old: i + 1 for i, old in enumerate(uniq)}
    out['id'] = out['id'].map(mapping).astype(int)
    return out, len(uniq)


def fit_one(df, cols, label, knots='K1', ci=True, time_xform=None):
    work = encode_admtype(df).copy()
    if time_xform is not None:
        work['time'] = time_xform(work['time'].values)

    feats = [c for c in cols if c != 'Adm_type'] + ADM_DUMMIES
    j = feats.index(ANCHOR)
    feats = [feats[j]] + [feats[k] for k in range(len(feats)) if k != j]

    work, N = remap_ids(work)
    work = work.sort_values(['id', 'time']).reset_index(drop=True)

    x      = work[feats].to_numpy(dtype=float)
    time_  = work['time'].to_numpy(dtype=float)
    delta  = work['delta'].to_numpy(dtype=int)
    id_vec = work['id'].to_numpy(dtype=int)

    n_evt = int((delta == 1).sum())
    n_cen = int((delta == 0).sum())
    print(f'\n=== LTM scenario: {label}  knots={knots}  ci={ci}  '
          f'(rows={len(work)}, patients={N}, '
          f'events={n_evt}, censors={n_cen}, '
          f'time range=[{time_.min():.4f}, {time_.max():.4f}]) ===')
    print(f'  anchor (beta_1=1): {feats[0]}')

    data = {'x': x, 'time': time_, 'delta': delta, 'id': id_vec}
    t0 = _time.time()
    est = fit(data, model='ltm', random_effect=True, knots=knots, ci=ci)
    elapsed = _time.time() - t0
    print(f'  total runtime: {elapsed:.1f}s  succ={est.success}')

    coef = est.beta.ravel()
    se   = est.se.ravel() if est.se is not None else np.full_like(coef, np.nan)
    z    = np.divide(coef, se, out=np.full_like(coef, np.nan),
                     where=(se > 0) & np.isfinite(se))
    pval = 2.0 * norm.sf(np.abs(z))
    lo95 = coef - 1.96 * se
    hi95 = coef + 1.96 * se

    s = pd.DataFrame({
        'feature': feats,
        'scenario': label,
        'coef': coef, 'se': se, 'z': z, 'p': pval,
        'lo95': lo95, 'hi95': hi95,
    })
    print(s.to_string(index=False, float_format=lambda v: f'{v:8.4f}'))
    return s, est


def main():
    df, cols = load_long()
    print(f'cohort_long_scaled : rows={len(df)}  '
          f'patients={df["id"].nunique()}  '
          f'events={int((df["delta"] == 1).sum())}  '
          f'censors={int((df["delta"] == 0).sum())}')

    # raw scenario (with CI from B=800 resampling — slow but tractable)
    s_raw, est_raw = fit_one(df, cols, 'raw', knots='K1', ci=True)

    # log10(1+t) for comparability with Cox log10 scenario
    s_log, est_log = fit_one(df, cols, 'log10', knots='K1', ci=True,
                              time_xform=lambda t: np.log10(1.0 + t))

    long = pd.concat([s_raw, s_log], ignore_index=True)
    long_path = os.path.join(OUT, 'ltm_coef_table.csv')
    long.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long.shape}')

    pivot = long.pivot(index='feature', columns='scenario',
                       values=['coef', 'se', 'p'])
    pivot.columns = [f'{m}_{s}' for m, s in pivot.columns]
    pivot = pivot.reindex(columns=[
        'coef_raw',   'se_raw',   'p_raw',
        'coef_log10', 'se_log10', 'p_log10'])
    pivot = pivot.reindex(s_raw['feature'].tolist())
    wide_path = os.path.join(OUT, 'ltm_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')

    # ---- functional parameters (for plotting) -----------------------
    def save_spline(est, label):
        raw = est.raw
        np.savez_compressed(
            os.path.join(OUT, f'ltm_spline_{label}.npz'),
            beta=est.beta, se=est.se,
            theta=raw['est_r'].ravel()[
                int(raw['p'].ravel()[0]):
                int(raw['p'].ravel()[0]) + int(raw['q_q'].ravel()[0])],
            alpha=raw['est_r'].ravel()[
                int(raw['p'].ravel()[0]) + int(raw['q_q'].ravel()[0]):],
            knots_0=raw['knots_0'].ravel(),
            knots_q=raw['knots_q'].ravel(),
            k0=int(raw['k0'].ravel()[0]),
            kq=int(raw['kq'].ravel()[0]),
            fish=raw.get('fish'),
        )
    save_spline(est_raw, 'raw')
    save_spline(est_log, 'log10')

    print('\n=== Significant features (LTM, Wald p < 0.05) ===')
    for sc in ['raw', 'log10']:
        sig = pivot[pivot[f'p_{sc}'] < 0.05].index.tolist()
        print(f'  {sc:5s}: {len(sig)} sig  -> {sig}')


if __name__ == '__main__':
    main()
