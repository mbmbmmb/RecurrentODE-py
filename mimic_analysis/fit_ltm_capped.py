"""Fit LTM (random-effect) on an admin-censored cohort.

Strategy: cap follow-up at CAP_DAYS instead of dropping patients.
  - event rows with time >= CAP are dropped
  - any patient whose follow-up exceeds CAP gets a censor row at time = CAP
    (delta = 0) so no patient is removed from the risk set
  - late-death patients (death after CAP) are reclassified to non-informative
    censoring at CAP

Two scenarios per cap: time_raw (days) and time_log10 = log10(1 + t).

Sign-correctness check: each fitted beta is compared against the
biologically-expected direction. Bicarbonate is the LTM identifiability
anchor (fixed at +1) so its expected sign is "anchor".
"""
from __future__ import annotations
import os, sys, time as _time, warnings, argparse
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

# Biologically-expected sign of beta for higher hazard = earlier readmission.
# '+' = higher value of feature -> higher hazard
# '-' = higher value of feature -> lower hazard (protective)
# '?' = ambiguous / non-monotone (don't penalize either sign)
# 'anchor' = LTM identifiability anchor (fixed at +1)
EXPECTED_SIGN = {
    'age':                 '+',   # older -> sicker
    'heartrate_mean':      '+',   # tachycardia
    'heartrate_spread':    '+',   # autonomic instability
    'sysbp_mean':          '-',   # higher BP protective in this acute window
    'sysbp_spread':        '+',
    'tempc_mean':          '?',   # both fever and hypothermia bad
    'tempc_spread':        '+',
    'PaO2FiO2_vent_min':   '-',   # higher = better oxygenation
    'urineoutput':         '-',   # higher = better renal/perfusion
    'bun_mean':            '+',   # azotemia
    'bun_spread':          '+',
    'wbc_mean':            '+',   # leukocytosis
    'wbc_spread':          '+',
    'potassium_mean':      '?',
    'potassium_spread':    '+',
    'sodium_mean':         '?',
    'sodium_spread':       '+',
    'bicarbonate_mean':    'anchor',
    'bicarbonate_spread':  '+',
    'bilirubin_mean':      '+',   # liver dysfunction
    'bilirubin_spread':    '+',
    'gcs_min':             '-',   # higher GCS = better neuro
    'AIDS':                '+',
    'HEM':                 '+',   # hematologic malignancy
    'METS':                '+',   # metastatic cancer
    'M_PaO2FiO2_vent_min': '+',   # tested -> sicker (informative missing)
    'M_bilirubin_mean':    '+',
    'Adm_type_scheduled':  '?',   # baseline reference depends on dummy code
    'Adm_type_unscheduled':'+',
}


def load_long():
    npz = np.load(NPZ, allow_pickle=True)
    cols = [c.decode() if isinstance(c, bytes) else str(c) for c in npz['x_cols']]
    df = pd.DataFrame(npz['x'], columns=cols)
    df['id']    = npz['id'].astype(int)
    df['time']  = npz['time'].astype(float)
    df['delta'] = npz['delta'].astype(int)
    return df, cols


def admin_censor(df: pd.DataFrame, cap_days: float) -> pd.DataFrame:
    """Apply administrative censoring at cap_days.

    Per patient:
      - keep all event rows with time < cap
      - if patient's max(time) <= cap: keep their original censor row
      - else: drop their original censor row + any events >= cap, and
        append a new censor row at time = cap, delta = 0
    """
    df = df.sort_values(['id', 'time']).reset_index(drop=True)
    fu_max = df.groupby('id')['time'].transform('max')
    capped_pat = fu_max > cap_days

    # Patients fully under the cap: keep all their rows.
    under = df[~capped_pat].copy()

    # Patients above the cap: keep events strictly before cap, drop the rest,
    # then append a synthetic censor at time = cap_days.
    over = df[capped_pat].copy()
    over_keep = over[(over['delta'] == 1) & (over['time'] < cap_days)].copy()

    # one synthetic censor row per over-cap patient. Reuse their X (it's
    # constant within id) by taking the first row.
    first_over = over.groupby('id', as_index=False).first()
    cens_new = first_over.copy()
    cens_new['time'] = float(cap_days)
    cens_new['delta'] = 0

    out = pd.concat([under, over_keep, cens_new], ignore_index=True)
    out = out.sort_values(['id', 'time', '-delta' if False else 'delta'],
                          ascending=[True, True, True]).reset_index(drop=True)
    # tie-break: events (delta=1) BEFORE censor (delta=0) at same time
    out['_sd'] = -out['delta']
    out = (out.sort_values(['id', 'time', '_sd'])
              .drop(columns='_sd').reset_index(drop=True))
    return out


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
    print(f'\n=== LTM {label}  knots={knots}  ci={ci}  '
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
    return s, est


def sign_check(s: pd.DataFrame, label: str, p_sig: float = 0.05) -> None:
    """Print sign-correctness table.

    Status:
      OK         expected sign matches fitted sign (or anchor / ambiguous)
      WRONG      expected sign present and fitted sign opposite
      WRONG-SIG  WRONG and Wald p < p_sig (worst case)
      n/a        ambiguous expected sign or anchor
    """
    rows = []
    n_ok = n_wrong = n_wrong_sig = n_na = 0
    for _, r in s.iterrows():
        f = r['feature']
        exp = EXPECTED_SIGN.get(f, '?')
        c = r['coef']; p = r['p']
        if exp == 'anchor':
            stat = 'anchor'; n_na += 1
        elif exp == '?':
            stat = 'n/a'; n_na += 1
        elif np.isnan(c):
            stat = 'nan'
        else:
            actual = '+' if c > 0 else '-'
            if actual == exp:
                stat = 'OK'; n_ok += 1
            else:
                if not np.isnan(p) and p < p_sig:
                    stat = 'WRONG-SIG'; n_wrong_sig += 1
                else:
                    stat = 'WRONG'; n_wrong += 1
        rows.append({'feature': f, 'expected': exp, 'coef': c,
                     'p': p, 'status': stat})
    rep = pd.DataFrame(rows)
    print(f'\n--- sign check [{label}]: '
          f'OK={n_ok}  WRONG-SIG={n_wrong_sig}  WRONG={n_wrong}  '
          f'n/a/anchor={n_na} ---')
    pd.set_option('display.max_rows', None)
    print(rep.to_string(index=False,
        formatters={'coef': lambda v: f'{v:8.3f}' if np.isfinite(v) else '   nan',
                    'p':    lambda v: f'{v:7.3g}' if np.isfinite(v) else '   nan'}))
    pd.reset_option('display.max_rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=float, default=1825.0,
                    help='cap days (admin-censor); default 5y = 1825')
    ap.add_argument('--tag', type=str, default=None,
                    help='filename suffix; default cap{int(cap)}d')
    args = ap.parse_args()

    cap = float(args.cap)
    tag = args.tag if args.tag is not None else f'cap{int(round(cap))}d'

    df, cols = load_long()
    n0_pat = df['id'].nunique()
    n0_row = len(df)
    n0_evt = int((df['delta'] == 1).sum())
    print(f'full   : rows={n0_row}  patients={n0_pat}  events={n0_evt}')

    df_cap = admin_censor(df, cap_days=cap)
    n1_pat = df_cap['id'].nunique()
    n1_row = len(df_cap)
    n1_evt = int((df_cap['delta'] == 1).sum())
    n1_cen = int((df_cap['delta'] == 0).sum())
    print(f'capped : rows={n1_row}  patients={n1_pat}  '
          f'events={n1_evt}  censors={n1_cen}  (cap = {cap:g} d)')
    print(f'  events kept: {n1_evt}/{n0_evt} = {n1_evt/n0_evt*100:.1f}%')
    print(f'  patients   : {n1_pat}/{n0_pat} (no patient dropped)')
    n_at_cap = int(((df_cap['delta'] == 0) & (df_cap['time'] == cap)).sum())
    print(f'  censor rows sitting AT cap = {cap:g}d : {n_at_cap}')

    # pass 1: point estimates without CI (full feature set)
    print('\n--- pass 1: point estimates (ci=False) ---')
    s_raw, _ = fit_one(df_cap, cols, 'raw',   knots='K1', ci=False)
    s_log, _ = fit_one(df_cap, cols, 'log10', knots='K1', ci=False,
                        time_xform=lambda t: np.log10(1.0 + t))

    # pass 2: with CI, sparse binaries dropped if needed
    DROP_SPARSE = ('AIDS', 'HEM', 'METS')
    print('\n--- pass 2: SE-bearing fit (ci=True, sparse binaries dropped) ---')

    def _safe(label, **kw):
        try:
            s, _ = fit_one(df_cap, cols, label, knots='K1', ci=True,
                            drop_features=DROP_SPARSE, **kw)
            return s
        except Exception as e:
            print(f'  pass2 {label} crashed: {type(e).__name__}: {e}')
            return None

    s_raw_ci = _safe('raw_ci')
    s_log_ci = _safe('log10_ci',
                      time_xform=lambda t: np.log10(1.0 + t))

    # sign checks
    print('\n' + '=' * 70)
    print(f' SIGN CORRECTNESS  (cap = {cap:g} days)')
    print('=' * 70)
    sign_check(s_raw, 'raw')
    sign_check(s_log, 'log10')
    if s_raw_ci is not None: sign_check(s_raw_ci, 'raw_ci')
    if s_log_ci is not None: sign_check(s_log_ci, 'log10_ci')

    # save
    parts = [s_raw, s_log] + [s for s in (s_raw_ci, s_log_ci) if s is not None]
    long_df = pd.concat(parts, ignore_index=True)
    long_path = os.path.join(OUT, f'ltm_{tag}_coef_table.csv')
    long_df.to_csv(long_path, index=False)
    print(f'\nwrote {long_path}: {long_df.shape}')

    pivot = long_df.pivot(index='feature', columns='scenario',
                          values=['coef', 'se', 'p'])
    pivot.columns = [f'{m}_{s}' for m, s in pivot.columns]
    pivot = pivot.reindex(s_raw['feature'].tolist())
    wide_path = os.path.join(OUT, f'ltm_{tag}_coef_wide.csv')
    pivot.to_csv(wide_path)
    print(f'wrote {wide_path}: {pivot.shape}')


if __name__ == '__main__':
    main()
