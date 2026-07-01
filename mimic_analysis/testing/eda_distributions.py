"""Diagnose distribution shape per feature and recommend a transform.

Per feature, plots a 2x2 grid of candidate transformations:

  [0] raw          [1] log(x)        (only if min > 0, else blank)
  [2] log1p(x)     [3] sqrt(|x|)

Every panel shows mean +/- 3 sigma cutoff lines computed AFTER the
transformation, and the post-transform skew in the title.

Outputs:
  merged_data/feature_dist_report.csv   per-feature stats + recommendation
  doc/plots/<col>.png                   4-panel comparison per feature
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'
PLOTS = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/doc/plots'

CONTINUOUS = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
]
SKIP = {'AIDS', 'HEM', 'METS', 'Adm_type'}


def recommend(col, x):
    """Return a row of per-column stats + recommended transform."""
    s = pd.Series(x).dropna()
    n_pos = int((s > 0).sum())
    n_zero = int((s == 0).sum())
    n_neg = int((s < 0).sum())
    if len(s) < 5:
        return None

    sk = float(stats.skew(s, bias=False))
    kt = float(stats.kurtosis(s, fisher=True, bias=False))
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()

    can_log = (s.min() >= 0)

    if abs(sk) < 0.5:
        rec = 'clip_3sigma'
        log_sk = None
    elif abs(sk) < 1.5:
        rec = ('clip_q99_right' if sk > 0 else 'clip_q01_left')
        log_sk = None
    else:
        if can_log:
            log_x = np.log1p(s)
            log_sk = float(stats.skew(log_x, bias=False))
            if abs(log_sk) < 0.75:
                rec = 'log1p_then_clip_3sigma'
            else:
                rec = ('clip_q99_right' if sk > 0 else 'clip_q01_left')
        else:
            log_sk = None
            rec = ('clip_q99_right' if sk > 0 else 'clip_q01_left')

    return {
        'col': col,
        'n': int(len(s)),
        'n_zero': n_zero,
        'n_neg': n_neg,
        'mean': mean, 'std': std,
        'p1': q[0.01], 'p5': q[0.05], 'p25': q[0.25],
        'median': q[0.5],
        'p75': q[0.75], 'p95': q[0.95], 'p99': q[0.99],
        'min': float(s.min()), 'max': float(s.max()),
        'skew': sk, 'log1p_skew': log_sk,
        'excess_kurtosis': kt,
        'recommend': rec,
    }


def _panel(ax, data, color, label):
    """Hist + mean +/- 3 sigma cutoff lines + skew/clipped count in title."""
    if data is None:
        ax.text(0.5, 0.5, f'{label}\nnot applicable',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return
    d = pd.Series(data).dropna().astype(float)
    if len(d) == 0:
        ax.text(0.5, 0.5, f'{label}\nempty', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return
    sk_pre = float(stats.skew(d, bias=False))
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    lo, hi = m - 3 * sd, m + 3 * sd
    d_clipped = d.clip(lo, hi)
    sk_post = float(stats.skew(d_clipped, bias=False))
    n_clip = int(((d < lo) | (d > hi)).sum())
    ax.hist(d_clipped, bins=80, color=color, alpha=0.85)
    ax.axvline(m, color='black', ls='-', lw=1, alpha=0.6,
               label=f'mean={m:.2f}')
    ax.axvline(lo, color='red', ls='--', lw=1,
               label=f'-3σ={lo:.2f}')
    ax.axvline(hi, color='red', ls='--', lw=1,
               label=f'+3σ={hi:.2f}')
    legend_extras = [
        f'skew_pre={sk_pre:+.2f}',
        f'skew_post={sk_post:+.2f}',
        f'clipped={n_clip}',
    ]
    for txt in legend_extras:
        ax.plot([], [], ' ', label=txt)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
    ax.set_title(f'{label}   skew_post = {sk_post:+.3f}',
                 fontsize=11, fontweight='bold')


def plot_one(col, x, info):
    """4-panel comparison: raw / log(x) / log1p / sqrt(|x|)."""
    s = pd.Series(x).dropna().astype(float)
    raw = s.values
    log_x = np.log(s.values) if s.min() > 0 else None
    log1p_x = np.log1p(s.values) if s.min() >= 0 else None
    sqrt_abs = np.sqrt(np.abs(s.values))

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    _panel(axes[0, 0], raw,      'steelblue',   'raw')
    _panel(axes[0, 1], log_x,    'orange',      'log(x)')
    _panel(axes[1, 0], log1p_x,  'seagreen',    'log1p(x)')
    _panel(axes[1, 1], sqrt_abs, 'mediumpurple', 'sqrt(|x|)')

    raw_min = float(s.min()); raw_max = float(s.max())
    raw_med = float(s.median()); raw_n = len(s)
    fig.suptitle(
        f'{col}   raw: n={raw_n}  min={raw_min:.3g}  '
        f'median={raw_med:.3g}  max={raw_max:.3g}   '
        f'(auto-rec: {info["recommend"]})',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(PLOTS, f'{col}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    df = pd.read_csv(os.path.join(OUT, 'cohort_clean.csv'))
    print(f'cohort_clean.csv : {df.shape}')

    rows = []
    for col in CONTINUOUS:
        if col in SKIP:
            continue
        info = recommend(col, df[col].values)
        if info is None:
            continue
        rows.append(info)
        plot_one(col, df[col].values, info)

    rep = pd.DataFrame(rows)
    cols_order = ['col', 'n', 'n_zero', 'n_neg', 'mean', 'std',
                  'min', 'p1', 'p5', 'p25', 'median', 'p75',
                  'p95', 'p99', 'max', 'skew', 'log1p_skew',
                  'excess_kurtosis', 'recommend']
    rep = rep[cols_order].sort_values('skew', key=lambda s: s.abs(),
                                      ascending=False)

    # Print compact summary
    print('\n=== Per-feature distribution & recommended transform ===')
    show = rep[['col', 'min', 'median', 'max', 'skew', 'log1p_skew',
                'recommend']].copy()
    for c in ('min', 'median', 'max'):
        show[c] = show[c].map(lambda v: f'{v:8.2f}')
    show['skew'] = show['skew'].map(lambda v: f'{v:+6.2f}')
    show['log1p_skew'] = show['log1p_skew'].map(
        lambda v: '   .  ' if pd.isna(v) else f'{v:+6.2f}')
    print(show.to_string(index=False))

    out = os.path.join(OUT, 'feature_dist_report.csv')
    rep.to_csv(out, index=False)
    print(f'\nwrote {out}')
    print(f'wrote {len(rows)} histogram PNGs to {PLOTS}')

    # Group counts
    print('\n=== Recommendation tally ===')
    print(rep['recommend'].value_counts().to_string())


if __name__ == '__main__':
    main()
