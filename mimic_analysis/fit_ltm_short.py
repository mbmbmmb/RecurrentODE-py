"""Refit random-effect LTM on the *short-term* sub-cohort.

Subsample rule: keep patients whose total follow-up (max(time) per id)
falls in **(1 day, 182.5 days)** = (24 h, half a year). Drops both the
near-zero censors (admin artifacts of same-day discharge / early
death) and the long-tail follow-up (>6 mo) that destabilises the
spline. Headline scenario is log10(1+t).
"""
from __future__ import annotations
import os, sys, time as _time, warnings
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

LO_DAYS = 0.0
HI_DAYS = 182.5
TAG = 'nolow'  # output suffix; was '' for the (1d, 0.5y) run


def load_long():
    npz = np.load(NPZ, allow_pickle=True)
    cols = [c.decode() if isinstance(c, bytes) else str(c) for c in npz['x_cols']]
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


def fit_one(df, cols, label, knots='K1', ci=True, time_xform=None,
            drop_features=()):
    work = encode_admtype(df).copy()
    if time_xform is not None:
        work['time'] = time_xform(work['time'].values)

    feats = [c for c in cols if c != 'Adm_type'] + ADM_DUMMIES
    feats = [f for f in feats if f not in drop_features]
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
    print(f'\n=== LTM short-term {label}  knots={knots}  ci={ci}  '
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
    s = pd.DataFrame({
        'feature': feats,
        'scenario': label,
        'coef': coef, 'se': se, 'z': z, 'p': pval,
        'lo95': coef - 1.96 * se, 'hi95': coef + 1.96 * se,
    })
    print(s.to_string(index=False, float_format=lambda v: f'{v:8.4f}'))
    return s, est


def main():
    df, cols = load_long()
    n0_pat = df['id'].nunique()
    n0_row = len(df)
    n0_evt = int((df['delta'] == 1).sum())

    # Per-patient total follow-up = max(time)
    fu = df.groupby('id')['time'].max()
    if LO_DAYS > 0:
        keep_ids = fu[(fu > LO_DAYS) & (fu < HI_DAYS)].index
    else:
        keep_ids = fu[fu < HI_DAYS].index
    df_sub = df[df['id'].isin(keep_ids)].copy()

    n1_pat = df_sub['id'].nunique()
    n1_row = len(df_sub)
    n1_evt = int((df_sub['delta'] == 1).sum())
    print(f'full   : rows={n0_row}  patients={n0_pat}  events={n0_evt}')
    print(f'subset : rows={n1_row}  patients={n1_pat}  events={n1_evt}  '
          f'(censor in ({LO_DAYS}, {HI_DAYS}) d)')
    print(f'kept patients: {n1_pat / n0_pat * 100:.1f}%   '
          f'kept events: {n1_evt / n0_evt * 100:.1f}%')

    # On this short-term subset (2,960 events), AIDS (22 evt), HEM (124),
    # and even METS (241) are sparse enough that B=800 resampling drives
    # the Fisher matrix singular. Two passes:
    #   pass 1: ci=False, full feature set, point estimates only.
    #   pass 2: ci=True with AIDS+HEM+METS dropped — usable SEs on the
    #           continuous + admin-type coefficients.
    DROP_SPARSE = ('AIDS', 'HEM', 'METS')

    print('\n--- pass 1: point estimates (ci=False) ---')
    s_raw, est_raw = fit_one(df_sub, cols, 'short_raw',   knots='K1', ci=False)
    s_log, est_log = fit_one(df_sub, cols, 'short_log10', knots='K1', ci=False,
                              time_xform=lambda t: np.log10(1.0 + t))

    print('\n--- pass 2: SE-bearing fit (ci=True, sparse binaries dropped) ---')
    def _safe_pass2(label, **kw):
        try:
            s, _ = fit_one(df_sub, cols, label, knots='K1', ci=True,
                            drop_features=DROP_SPARSE, **kw)
            return s
        except Exception as e:
            print(f'  pass2 {label} crashed: {type(e).__name__}: {e}')
            return None

    s_raw_ci = _safe_pass2('short_raw_ci')
    s_log_ci = _safe_pass2('short_log10_ci',
                            time_xform=lambda t: np.log10(1.0 + t))

    parts = [s_raw, s_log] + [s for s in (s_raw_ci, s_log_ci) if s is not None]
    long = pd.concat(parts, ignore_index=True)
    suffix = f'_{TAG}' if TAG else ''
    long_path = os.path.join(OUT, f'ltm_short{suffix}_coef_table.csv')
    long.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long.shape}')

    pivot = long.pivot(index='feature', columns='scenario',
                       values=['coef', 'se', 'p'])
    pivot.columns = [f'{m}_{s}' for m, s in pivot.columns]
    pivot = pivot.reindex(columns=[
        'coef_short_raw',     'se_short_raw',     'p_short_raw',
        'coef_short_log10',   'se_short_log10',   'p_short_log10',
        'coef_short_raw_ci',  'se_short_raw_ci',  'p_short_raw_ci',
        'coef_short_log10_ci','se_short_log10_ci','p_short_log10_ci',
    ])
    pivot = pivot.reindex(s_raw['feature'].tolist())
    wide_path = os.path.join(OUT, f'ltm_short{suffix}_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')

    print('\n=== Significant features (LTM short-term, Wald p < 0.05) ===')
    for sc in ['short_raw_ci', 'short_log10_ci']:
        col = f'p_{sc}'
        sig = pivot[pivot[col].fillna(1) < 0.05].index.tolist()
        print(f'  {sc:16s}: {len(sig)} sig  -> {sig}')


if __name__ == '__main__':
    main()
