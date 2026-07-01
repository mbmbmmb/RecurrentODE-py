"""LTM RE on full cohort with heartrate_mean as anchor (beta_1 = 1).

The LTM MLE in RecurrentODE_py/random_effect/ltm/mle.py initializes its
beta from a Cox fit (cox_rec) internally and normalizes by the first
feature's coefficient. So pinning beta_1 = 1 on a particular feature is
done by placing that feature first in the column order. The Cox-based
initialization happens automatically.

heartrate_mean is chosen because Cox PH gives it a clean, sign-correct
significant coefficient (β=+0.058, p=6.7e-04) — a sensible scale anchor.
"""
from __future__ import annotations
import os, sys, time as _time, warnings
import numpy as np, pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ  = os.path.join(ROOT, 'merged_data/cohort_long_scaled.npz')
OUT  = os.path.join(ROOT, 'merged_data')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recurrent_ode import fit  # noqa: E402

ADM_DUMMIES = ['Adm_type_scheduled', 'Adm_type_unscheduled']
ANCHOR = 'heartrate_mean'

EXPECTED_SIGN = {
    'age':'+','heartrate_mean':'anchor','heartrate_spread':'+','sysbp_mean':'-',
    'sysbp_spread':'+','tempc_mean':'?','tempc_spread':'+',
    'PaO2FiO2_vent_min':'-','urineoutput':'-','bun_mean':'+','bun_spread':'+',
    'wbc_mean':'+','wbc_spread':'+','potassium_mean':'?','potassium_spread':'+',
    'sodium_mean':'?','sodium_spread':'+','bicarbonate_mean':'?',
    'bicarbonate_spread':'+','bilirubin_mean':'+','bilirubin_spread':'+',
    'gcs_min':'-','AIDS':'+','HEM':'+','METS':'+',
    'M_PaO2FiO2_vent_min':'+','M_bilirubin_mean':'+',
    'Adm_type_scheduled':'?','Adm_type_unscheduled':'+',
}


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


def fit_one(df, cols, label, time_xform=None):
    work = encode_admtype(df).copy()
    if time_xform is not None:
        work['time'] = time_xform(work['time'].values)

    feats = [c for c in cols if c != 'Adm_type'] + ADM_DUMMIES
    j = feats.index(ANCHOR)
    feats = [feats[j]] + [feats[k] for k in range(len(feats)) if k != j]

    work, N = remap_ids(work)
    work = work.sort_values(['id', 'time']).reset_index(drop=True)
    x = work[feats].to_numpy(float)
    time_ = work['time'].to_numpy(float)
    delta = work['delta'].to_numpy(int)
    id_v  = work['id'].to_numpy(int)

    n_evt = int((delta == 1).sum())
    n_cen = int((delta == 0).sum())
    print(f'\n=== LTM {label}  (rows={len(work)}, patients={N}, '
          f'events={n_evt}, censors={n_cen}, '
          f'time range=[{time_.min():.4f}, {time_.max():.4f}]) ===')
    print(f'  anchor (beta_1=1): {feats[0]}  (Cox init auto via cox_rec)')

    t0 = _time.time()
    est = fit({'x': x, 'time': time_, 'delta': delta, 'id': id_v},
              model='ltm', random_effect=True, knots='K1', ci=True)
    print(f'  total runtime: {_time.time()-t0:.1f}s  succ={est.success}')

    coef = est.beta.ravel()
    se   = est.se.ravel() if est.se is not None else np.full_like(coef, np.nan)
    z = np.divide(coef, se, out=np.full_like(coef, np.nan),
                  where=(se > 0) & np.isfinite(se))
    p = 2.0 * norm.sf(np.abs(z))
    s = pd.DataFrame({'feature': feats, 'scenario': label,
                      'coef': coef, 'se': se, 'z': z, 'p': p,
                      'lo95': coef - 1.96*se, 'hi95': coef + 1.96*se})
    return s, est


def sign_check(s, label):
    rep = s.copy()
    rep['expected'] = rep.feature.map(EXPECTED_SIGN)
    rep['actual']   = np.where(rep.coef > 0, '+', '-')
    def st(r):
        if r['expected'] == 'anchor': return 'anchor'
        if r['expected'] == '?':      return 'n/a'
        if not np.isfinite(r['coef']): return 'nan'
        return 'OK' if r['actual'] == r['expected'] else 'WRONG'
    rep['status'] = rep.apply(st, axis=1)
    sig = rep.dropna(subset=['p']).query('p<0.05').sort_values('p')
    n_ok = (rep.status == 'OK').sum()
    n_wr = (rep.status == 'WRONG').sum()
    n_sok = (sig.status == 'OK').sum()
    n_swr = (sig.status == 'WRONG').sum()
    n_samb = sig.status.isin(['n/a', 'anchor']).sum()
    print(f'\n--- sign check [{label}]: '
          f'OK(all)={n_ok}  WRONG(all)={n_wr}    '
          f'sig p<0.05: n={len(sig)} (OK={n_sok}, WRONG={n_swr}, '
          f'ambig={n_samb}) ---')
    if len(sig):
        print(sig[['feature','expected','actual','coef','se','p','status']]
              .to_string(index=False,
                formatters={'coef':lambda v:f'{v:+7.3f}',
                            'se':  lambda v:f'{v:6.3f}',
                            'p':   lambda v:f'{v:9.2e}'}))
    print('\n  Full coefficient table:')
    print(rep[['feature','expected','coef','p','status']]
          .to_string(index=False,
            formatters={'coef':lambda v:f'{v:+7.3f}' if np.isfinite(v) else '   nan',
                        'p':   lambda v:f'{v:9.2e}' if np.isfinite(v) else '   nan'}))
    return rep, sig


def main():
    df, cols = load_long()
    print(f'cohort: rows={len(df)} pat={df.id.nunique()} '
          f'events={int((df.delta==1).sum())}')

    s_log, _ = fit_one(df, cols, 'hr_log10',
                        time_xform=lambda t: np.log10(1.0 + t))
    s_raw, _ = fit_one(df, cols, 'hr_raw')

    print('\n' + '='*70)
    print(' SIGN CORRECTNESS  (anchor = heartrate_mean, no cap, full cohort)')
    print('='*70)
    sign_check(s_log, 'hr_log10')
    sign_check(s_raw, 'hr_raw')

    long_df = pd.concat([s_log, s_raw], ignore_index=True)
    long_path = os.path.join(OUT, 'ltm_hr_anchor_coef_table.csv')
    long_df.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long_df.shape}')

    pivot = long_df.pivot(index='feature', columns='scenario',
                          values=['coef','se','p'])
    pivot.columns = [f'{m}_{s}' for m,s in pivot.columns]
    pivot = pivot.reindex(s_log.feature.tolist())
    wide_path = os.path.join(OUT, 'ltm_hr_anchor_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')


if __name__ == '__main__':
    main()
