"""Refit Cox-RE and LTM-RE on the full cohort with NO spread features.

Drops every *_spread column so only the per-stay mean (which is the
midpoint of (min, max)) is retained. Existing *_min features
(PaO2FiO2_vent_min, gcs_min) and the demographics / indicators /
Adm_type dummies are kept.

Two model fits are produced:
  1. Cox-RE  (raw, log10(1+t))           --  cox_nospread_*
  2. LTM-RE  (hr_log10, anchor=hr_mean)  --  ltm_nospread_hr_log10_*

Outputs land in merged_data/ with the `nospread` tag.
"""
from __future__ import annotations
import os, sys, time as _time, warnings, argparse
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

# Sign expectations (no spread features so spread entries are gone)
EXPECTED_SIGN = {
    'age':'+','heartrate_mean':'?','sysbp_mean':'-','tempc_mean':'?',
    'PaO2FiO2_vent_min':'-','urineoutput':'-','bun_mean':'+',
    'wbc_mean':'+','potassium_mean':'?','sodium_mean':'?',
    'bicarbonate_mean':'?','bilirubin_mean':'+','gcs_min':'-',
    'AIDS':'+','HEM':'+','METS':'+',
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


def admin_censor(df: pd.DataFrame, cap_days: float) -> pd.DataFrame:
    """Cap follow-up at cap_days. Mirrors fit_ltm_capped.admin_censor.

    Per patient: keep events with time<cap; if max(time)>cap, drop the
    original censor + late events and append a synthetic censor at
    time=cap, delta=0. No patient is removed.
    """
    df = df.sort_values(['id', 'time']).reset_index(drop=True)
    fu_max = df.groupby('id')['time'].transform('max')
    capped_pat = fu_max > cap_days
    under = df[~capped_pat].copy()
    over = df[capped_pat].copy()
    over_keep = over[(over['delta'] == 1) & (over['time'] < cap_days)].copy()
    first_over = over.groupby('id', as_index=False).first()
    cens_new = first_over.copy()
    cens_new['time']  = float(cap_days)
    cens_new['delta'] = 0
    out = pd.concat([under, over_keep, cens_new], ignore_index=True)
    out['_sd'] = -out['delta']
    out = (out.sort_values(['id', 'time', '_sd'])
              .drop(columns='_sd').reset_index(drop=True))
    return out


def remap_ids(df):
    out = df.copy()
    uniq = np.sort(out['id'].unique())
    mapping = {old: i + 1 for i, old in enumerate(uniq)}
    out['id'] = out['id'].map(mapping).astype(int)
    return out, len(uniq)


def select_nospread_feats(cols):
    """Drop every *_spread column. Drop the raw Adm_type (replaced by dummies).
    Anchor (heartrate_mean) is placed first only when this is used for LTM."""
    keep = [c for c in cols
            if (not c.endswith('_spread')) and c != 'Adm_type']
    return keep + ADM_DUMMIES


def _build_xframe(df, cols, anchor_first=False, time_xform=None):
    work = encode_admtype(df).copy()
    if time_xform is not None:
        work['time'] = time_xform(work['time'].values)
    feats = select_nospread_feats(cols)
    if anchor_first and ANCHOR in feats:
        j = feats.index(ANCHOR)
        feats = [feats[j]] + [feats[k] for k in range(len(feats)) if k != j]
    work, N = remap_ids(work)
    work = work.sort_values(['id', 'time']).reset_index(drop=True)
    x = work[feats].to_numpy(float)
    return work, N, feats, x


def fit_cox(df, cols, label, time_xform=None):
    work, N, feats, x = _build_xframe(df, cols, anchor_first=False,
                                      time_xform=time_xform)
    time_ = work['time'].to_numpy(float)
    delta = work['delta'].to_numpy(int)
    id_v  = work['id'].to_numpy(int)
    n_evt = int((delta == 1).sum())
    n_cen = int((delta == 0).sum())
    print(f'\n=== Cox {label}  (rows={len(work)}, patients={N}, '
          f'events={n_evt}, censors={n_cen}, '
          f'time range=[{time_.min():.4f}, {time_.max():.4f}]) ===')
    print(f'  feature count: {len(feats)}  (no *_spread)')
    t0 = _time.time()
    est = fit({'x': x, 'time': time_, 'delta': delta, 'id': id_v},
              model='cox', random_effect=True, ci=True)
    print(f'  total runtime: {_time.time()-t0:.1f}s')
    coef = est.beta.ravel(); se = est.se.ravel()
    z = coef/se; p = 2.0*norm.sf(np.abs(z))
    return pd.DataFrame({'feature': feats, 'scenario': f'cox_{label}',
                         'coef': coef, 'se': se, 'z': z, 'p': p,
                         'lo95': coef-1.96*se, 'hi95': coef+1.96*se})


def fit_ltm(df, cols, label, time_xform=None):
    work, N, feats, x = _build_xframe(df, cols, anchor_first=True,
                                      time_xform=time_xform)
    time_ = work['time'].to_numpy(float)
    delta = work['delta'].to_numpy(int)
    id_v  = work['id'].to_numpy(int)
    n_evt = int((delta == 1).sum())
    n_cen = int((delta == 0).sum())
    print(f'\n=== LTM {label}  (rows={len(work)}, patients={N}, '
          f'events={n_evt}, censors={n_cen}, '
          f'time range=[{time_.min():.4f}, {time_.max():.4f}]) ===')
    print(f'  feature count: {len(feats)}  (no *_spread)')
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
    return pd.DataFrame({'feature': feats, 'scenario': f'ltm_{label}',
                         'coef': coef, 'se': se, 'z': z, 'p': p,
                         'lo95': coef-1.96*se, 'hi95': coef+1.96*se})


def sign_check(s, label):
    rep = s.copy()
    rep['expected'] = rep.feature.map(EXPECTED_SIGN).fillna('?')
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=None,
                    help='admin-censor at cap days (e.g. 182.5 = 0.5y)')
    ap.add_argument('--tag', type=str, default=None,
                    help='filename suffix; default "" or cap{int(cap)}d')
    args = ap.parse_args()
    cap = args.cap
    tag = args.tag if args.tag is not None else (
        '' if cap is None else f'_cap{int(round(cap))}d')

    df, cols = load_long()
    print(f'full   : rows={len(df)} pat={df.id.nunique()} '
          f'events={int((df.delta==1).sum())}')

    if cap is not None:
        df = admin_censor(df, cap_days=cap)
        n_evt = int((df.delta==1).sum())
        n_cen = int((df.delta==0).sum())
        print(f'capped : rows={len(df)} pat={df.id.nunique()} '
              f'events={n_evt} censors={n_cen}  (cap={cap:g}d)')

    kept = select_nospread_feats(cols)
    dropped = [c for c in cols if c.endswith('_spread')]
    print(f'\ndropped *_spread features ({len(dropped)}): {dropped}')
    print(f'kept features ({len(kept)}): {kept}')

    # ---- Cox: raw + log10 ------------------------------------------------
    s_cox_raw = fit_cox(df, cols, 'raw')
    s_cox_log = fit_cox(df, cols, 'log10',
                        time_xform=lambda t: np.log10(1.0 + t))

    # ---- LTM: hr_log10 (anchor heartrate_mean = +1) ----------------------
    s_ltm_log = fit_ltm(df, cols, 'hr_log10',
                        time_xform=lambda t: np.log10(1.0 + t))

    # ---- Sign reports ----------------------------------------------------
    print('\n' + '='*70)
    print(' SIGN CORRECTNESS  (no spread features, full cohort)')
    print('='*70)
    sign_check(s_cox_raw, 'cox_raw')
    sign_check(s_cox_log, 'cox_log10')
    sign_check(s_ltm_log, 'ltm_hr_log10')

    # ---- Persist ---------------------------------------------------------
    long_df = pd.concat([s_cox_raw, s_cox_log, s_ltm_log], ignore_index=True)
    long_path = os.path.join(OUT, f'nospread{tag}_coef_table.csv')
    long_df.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long_df.shape}')

    pivot = long_df.pivot(index='feature', columns='scenario',
                          values=['coef','se','p'])
    pivot.columns = [f'{m}_{s}' for m,s in pivot.columns]
    feat_order = s_cox_raw['feature'].tolist()
    pivot = pivot.reindex(feat_order)
    wide_path = os.path.join(OUT, f'nospread{tag}_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')


if __name__ == '__main__':
    main()
