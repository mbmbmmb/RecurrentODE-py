"""Re-fit random-effect LTM after dropping all death-censored patients.

Diagnostic for the persistent wrong-sign cluster. If the wrong signs are
driven by informative censoring (death is correlated with covariates),
removing the 4,712 death-censored patients should flip those signs
toward biological expectation.

Two scenarios: log10(1+t) and raw t, both no-cap. ALL features kept.
"""
from __future__ import annotations
import os, sys, time as _time, warnings
import numpy as np, pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ  = os.path.join(ROOT, 'merged_data/cohort_long_scaled.npz')
CSV  = os.path.join(ROOT, 'merged_data/cohort_long.csv')
OUT  = os.path.join(ROOT, 'merged_data')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recurrent_ode import fit  # noqa: E402

ADM_DUMMIES = ['Adm_type_scheduled', 'Adm_type_unscheduled']
ANCHOR = 'bicarbonate_mean'

EXPECTED_SIGN = {
    'age':'+','heartrate_mean':'+','heartrate_spread':'+','sysbp_mean':'-',
    'sysbp_spread':'+','tempc_mean':'?','tempc_spread':'+',
    'PaO2FiO2_vent_min':'-','urineoutput':'-','bun_mean':'+','bun_spread':'+',
    'wbc_mean':'+','wbc_spread':'+','potassium_mean':'?','potassium_spread':'+',
    'sodium_mean':'?','sodium_spread':'+','bicarbonate_mean':'anchor',
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


def drop_death_censored(df):
    """Drop all rows whose patient was death-censored."""
    csv = pd.read_csv(CSV, usecols=['id', 'is_death_censored'])
    dc_ids = set(csv.loc[csv.is_death_censored == 1, 'id'].unique())
    keep = ~df['id'].isin(dc_ids)
    return df[keep].copy(), len(dc_ids)


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
    print(f'\n=== LTM {label}  ci={ci}  '
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
        'feature': feats, 'scenario': label,
        'coef': coef, 'se': se, 'z': z, 'p': pval,
        'lo95': coef - 1.96*se, 'hi95': coef + 1.96*se,
    })
    return s, est


def sign_check(s, label):
    rep = s.copy()
    rep['expected'] = rep['feature'].map(EXPECTED_SIGN)
    rep['actual']   = np.where(rep['coef']>0,'+','-')
    def st(r):
        if r['expected']=='anchor': return 'anchor'
        if r['expected']=='?':      return 'n/a'
        if not np.isfinite(r['coef']): return 'nan'
        return 'OK' if r['actual']==r['expected'] else 'WRONG'
    rep['status'] = rep.apply(st, axis=1)
    n_ok=(rep.status=='OK').sum()
    n_wr=(rep.status=='WRONG').sum()
    print(f'\n--- sign check [{label}]: OK={n_ok}  WRONG={n_wr} ---')
    pd.set_option('display.max_rows', None)
    print(rep[['feature','expected','coef','p','status']].to_string(
        index=False,
        formatters={'coef':lambda v:f'{v:+8.3f}' if np.isfinite(v) else '   nan',
                    'p':   lambda v:f'{v:9.2e}' if np.isfinite(v) else '   nan'}))
    pd.reset_option('display.max_rows')

    sig = rep.dropna(subset=['p']).query('p<0.05').sort_values('p')
    n_sig=len(sig)
    n_sig_ok=(sig.status=='OK').sum()
    n_sig_wr=(sig.status=='WRONG').sum()
    print(f'\n  Significant (p<0.05) [{label}]: '
          f'n={n_sig}  OK={n_sig_ok}  WRONG={n_sig_wr}')
    if n_sig:
        print(sig[['feature','expected','actual','coef','se','p','status']]
            .to_string(index=False,
            formatters={'coef':lambda v:f'{v:+7.3f}',
                        'se':  lambda v:f'{v:6.3f}',
                        'p':   lambda v:f'{v:9.2e}'}))
    return rep, sig


def main():
    df, cols = load_long()
    n0_pat = df['id'].nunique()
    n0_row = len(df)
    n0_evt = int((df['delta']==1).sum())
    print(f'full cohort: rows={n0_row}  pat={n0_pat}  events={n0_evt}')

    df_nd, n_drop = drop_death_censored(df)
    n1_pat = df_nd['id'].nunique()
    n1_row = len(df_nd)
    n1_evt = int((df_nd['delta']==1).sum())
    print(f'dropped death-cens: {n_drop} patients')
    print(f'remaining cohort  : rows={n1_row}  pat={n1_pat}  events={n1_evt}')
    print(f'  events kept: {n1_evt}/{n0_evt} = {n1_evt/n0_evt*100:.1f}%')

    s_log, _ = fit_one(df_nd, cols, 'nodeath_log10', ci=True,
                        time_xform=lambda t: np.log10(1.0+t))
    s_raw, _ = fit_one(df_nd, cols, 'nodeath_raw',   ci=True)

    print('\n' + '='*70)
    print(' SIGN CORRECTNESS  (death-censored DROPPED)')
    print('='*70)
    rep_log, sig_log = sign_check(s_log, 'nodeath_log10')
    rep_raw, sig_raw = sign_check(s_raw, 'nodeath_raw')

    long_df = pd.concat([s_log, s_raw], ignore_index=True)
    long_path = os.path.join(OUT, 'ltm_nodeath_coef_table.csv')
    long_df.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long_df.shape}')

    pivot = long_df.pivot(index='feature', columns='scenario',
                          values=['coef','se','p'])
    pivot.columns = [f'{m}_{s}' for m,s in pivot.columns]
    pivot = pivot.reindex(s_log['feature'].tolist())
    wide_path = os.path.join(OUT, 'ltm_nodeath_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')


if __name__ == '__main__':
    main()
