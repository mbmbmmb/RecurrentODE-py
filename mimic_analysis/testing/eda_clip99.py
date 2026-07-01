"""Per-feature histogram of raw data with middle-99% (q0.5/q99.5) clip bounds.

For each continuous feature (loaded from the unclipped MIMIC input.csv,
joined with valid_aids.csv for the cohort), this script:
  - Computes summary stats: n, min, max, mean, std on the raw values.
  - Computes middle-99% clipping bounds: q0.005 (lo) and q0.995 (hi).
  - Plots a histogram of the RAW data with two red dashed vertical lines
    at lo / hi, and a legend showing min/max/mean/std + lo/hi.
  - Saves to doc/plots/clip99/<col>.png.

A summary CSV is written to merged_data/clip99_bounds.csv with one row
per feature listing all stats + clip bounds + count clipped on each side.

No transformation is applied; this is the raw-data view used to decide
which clipping strategy to commit to.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/MIMIC Data'
SRC = os.path.join(ROOT, 'processed-data')
OUT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/merged_data'
PLOTS = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis/doc/plots/clip99'

V_NAMES = [
    'age', 'heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
    'tempc_max', 'tempc_min', 'PaO2FiO2_vent_min', 'urineoutput',
    'bun_min', 'bun_max', 'wbc_min', 'wbc_max', 'potassium_min',
    'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min',
    'bicarbonate_max', 'bilirubin_min', 'bilirubin_max', 'gcs_min',
    'AIDS', 'HEM', 'METS', 'Adm_type',
]
SKIP = {'AIDS', 'HEM', 'METS', 'Adm_type'}

# Per-feature legend placement override (default = 'upper right').
LEGEND_LOC = {
    'tempc_min':  'upper left',
    'sodium_min': 'upper left',
    'gcs_min':    'upper center',
}


def load_raw():
    inp = pd.read_csv(os.path.join(SRC, 'input.csv'), header=None,
                      names=V_NAMES)
    aids = pd.read_csv(os.path.join(SRC, 'valid_aids.csv'), header=None,
                       names=['HADM_ID']).astype({'HADM_ID': 'int64'})
    df = pd.concat([aids, inp], axis=1)
    # Cap age at 90 (MIMIC adds ~300 yr offset for >=89). Done here so
    # the histogram isn't dominated by the 300-yr blob; this is the
    # only adjustment to "raw" we apply.
    df.loc[df['age'] > 90, 'age'] = 90
    return df


def plot_one(col, x):
    s = pd.Series(x).dropna().astype(float)
    n = len(s)
    mn, mx = float(s.min()), float(s.max())
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    lo = float(s.quantile(0.005))
    hi = float(s.quantile(0.995))
    n_below = int((s < lo).sum())
    n_above = int((s > hi).sum())
    s_clipped = s.clip(lower=lo, upper=hi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(s_clipped, bins=80, color='steelblue', alpha=0.85)
    ax.axvline(lo, color='red', ls='--', lw=1.4)
    ax.axvline(hi, color='red', ls='--', lw=1.4)

    handles = [
        plt.Line2D([], [], color='none', label=f'n          = {n}'),
        plt.Line2D([], [], color='none', label=f'raw min    = {mn:.3g}'),
        plt.Line2D([], [], color='none', label=f'raw max    = {mx:.3g}'),
        plt.Line2D([], [], color='none', label=f'raw mean   = {mu:.3g}'),
        plt.Line2D([], [], color='none', label=f'raw std    = {sd:.3g}'),
        plt.Line2D([], [], color='red', ls='--', lw=1.4,
                   label=f'q0.5%      = {lo:.3g}'),
        plt.Line2D([], [], color='red', ls='--', lw=1.4,
                   label=f'q99.5%     = {hi:.3g}'),
        plt.Line2D([], [], color='none',
                   label=f'clipped: {n_below} lo / {n_above} hi'),
    ]
    ax.legend(handles=handles, loc=LEGEND_LOC.get(col, 'upper right'),
              fontsize=9, framealpha=0.9, handlelength=1.4)
    ax.set_title(
        f'{col}   distribution after middle-99% clip (raw stats in legend)',
        fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'{col}.png'), dpi=110)
    plt.close(fig)

    return {
        'col': col, 'n': n,
        'min': mn, 'max': mx, 'mean': mu, 'std': sd,
        'q0.5%': lo, 'q99.5%': hi,
        'n_below_lo': n_below, 'n_above_hi': n_above,
        'pct_clipped': 100 * (n_below + n_above) / n,
    }


def plot_categorical(col, x):
    """Bar chart of value counts for a categorical/binary feature."""
    s = pd.Series(x).dropna()
    n = len(s)
    counts = s.value_counts().sort_index()
    cats = counts.index.astype(int).tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([str(c) for c in cats], counts.values,
                  color='steelblue', alpha=0.85)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f'{int(v)}\n({100 * v / n:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax.set_title(f'{col}   value counts (n = {n})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('count')
    ax.margins(y=0.18)
    handles = [
        plt.Line2D([], [], color='none', label=f'n      = {n}'),
        plt.Line2D([], [], color='none',
                   label=f'levels = {len(cats)}'),
        plt.Line2D([], [], color='none',
                   label=f'values = {sorted(cats)}'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              framealpha=0.9, handlelength=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'{col}.png'), dpi=110)
    plt.close(fig)


def plot_corr_heatmap(df_clipped, cols, out_name, title):
    """Pearson correlation heatmap of post-clip + 0-filled features."""
    corr = df_clipped[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2, 0.55 * n + 1))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6,
                    color='white' if abs(v) > 0.55 else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    out = os.path.join(PLOTS, out_name)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return corr


# Lab pairs collapsed to (mean, spread) in clean_cohort.py. Includes
# the six high-corr (>0.8) pairs plus heartrate / sysbp (extended on
# user request even though their post-clip |r| was below 0.8).
MEAN_SPREAD_PAIRS = [
    'tempc', 'bilirubin', 'bun', 'sodium', 'bicarbonate', 'wbc',
    'heartrate', 'sysbp',
]


def build_meanspread_frame(df_raw):
    """Replace highly-correlated *_min / *_max pairs with mean/spread.

    For each pair, mean = (max + min) / 2, spread = max - min, computed
    from RAW values. Other features keep their original min/max.
    """
    out = df_raw.copy()
    cols_new = []
    for base in MEAN_SPREAD_PAIRS:
        lo_col, hi_col = f'{base}_min', f'{base}_max'
        out[f'{base}_mean']   = (out[hi_col] + out[lo_col]) / 2.0
        out[f'{base}_spread'] = (out[hi_col] - out[lo_col]).abs()
        out = out.drop(columns=[lo_col, hi_col])
        cols_new.extend([f'{base}_mean', f'{base}_spread'])
    keep = [c for c in V_NAMES if c not in SKIP
            and not any(c == f'{b}_min' or c == f'{b}_max'
                        for b in MEAN_SPREAD_PAIRS)]
    return out, keep + cols_new


def clip99_fill0(df_raw, cols):
    """Apply per-column middle-99% clip then fill NaN with 0."""
    out = df_raw.copy()
    for col in cols:
        s = out[col]
        lo, hi = float(s.quantile(0.005)), float(s.quantile(0.995))
        out[col] = s.clip(lower=lo, upper=hi).fillna(0.0)
    return out


def main():
    os.makedirs(PLOTS, exist_ok=True)
    df = load_raw()
    print(f'raw input: {df.shape}')

    cont = [c for c in V_NAMES if c not in SKIP]
    rows = []
    for col in cont:
        rows.append(plot_one(col, df[col].values))
    rep = pd.DataFrame(rows)

    # Categorical / binary features (no clip, no quantile lines).
    cat_cols = [c for c in V_NAMES if c in SKIP]
    for col in cat_cols:
        plot_categorical(col, df[col].values)
    print(f'wrote {len(cat_cols)} categorical bar charts to {PLOTS}')

    show = rep.copy()
    for c in ('min', 'max', 'mean', 'std', 'q0.5%', 'q99.5%'):
        show[c] = show[c].map(lambda v: f'{v:9.3g}')
    show['pct_clipped'] = show['pct_clipped'].map(lambda v: f'{v:5.2f}%')
    print('\n=== Per-feature summary stats + middle-99% clip bounds ===')
    print(show.to_string(index=False))

    out = os.path.join(OUT, 'clip99_bounds.csv')
    rep.to_csv(out, index=False)
    print(f'\nwrote {out}')
    print(f'wrote {len(rows)} histogram PNGs to {PLOTS}')

    # ---- Post-clip correlation heatmap (raw min/max layout) --------
    print('\n=== Post-clip correlation (NaN -> 0) ===')
    df_clipped = clip99_fill0(df, cont)
    corr = plot_corr_heatmap(
        df_clipped, cont,
        out_name='correlation.png',
        title='Pearson correlation: post-clip features (NaN -> 0)')
    corr_path = os.path.join(OUT, 'clip99_correlation.csv')
    corr.to_csv(corr_path)
    print(f'wrote {corr_path}: {corr.shape}')

    cm = corr.values.copy()
    np.fill_diagonal(cm, 0.0)
    pairs = sorted(
        ((cont[i], cont[j], cm[i, j])
         for i in range(len(cont)) for j in range(i + 1, len(cont))),
        key=lambda t: abs(t[2]), reverse=True)
    print('\ntop 15 |corr| pairs:')
    for a, b, v in pairs[:15]:
        print(f'  {a:22s}  {b:22s}  {v:+.3f}')

    # ---- Mean/spread re-parameterization ----------------------------
    print('\n=== Mean/spread re-parameterization '
          f'(replaces {len(MEAN_SPREAD_PAIRS)} min/max pairs) ===')
    df_ms_raw, cols_ms = build_meanspread_frame(df)

    # Per-feature hist for each new (mean, spread) column.
    print('\n=== Mean/spread feature histograms ===')
    new_cols = [c for c in cols_ms if c.endswith('_mean')
                or c.endswith('_spread')]
    for col in new_cols:
        plot_one(col, df_ms_raw[col].values)
    print(f'wrote {len(new_cols)} mean/spread histogram PNGs to {PLOTS}')

    df_ms = clip99_fill0(df_ms_raw, cols_ms)
    corr_ms = plot_corr_heatmap(
        df_ms, cols_ms,
        out_name='correlation_meanspread.png',
        title='Pearson correlation: mean/spread re-param (NaN -> 0)')
    ms_path = os.path.join(OUT, 'clip99_correlation_meanspread.csv')
    corr_ms.to_csv(ms_path)
    print(f'wrote {ms_path}: {corr_ms.shape}')

    cm_ms = corr_ms.values.copy()
    np.fill_diagonal(cm_ms, 0.0)
    pairs_ms = sorted(
        ((cols_ms[i], cols_ms[j], cm_ms[i, j])
         for i in range(len(cols_ms)) for j in range(i + 1, len(cols_ms))),
        key=lambda t: abs(t[2]), reverse=True)
    print('\ntop 15 |corr| pairs (mean/spread):')
    for a, b, v in pairs_ms[:15]:
        print(f'  {a:22s}  {b:22s}  {v:+.3f}')

    n_high_before = sum(1 for *_, v in pairs if abs(v) > 0.8)
    n_high_after = sum(1 for *_, v in pairs_ms if abs(v) > 0.8)
    print(f'\n|corr| > 0.8 pair count: '
          f'{n_high_before} (min/max)  ->  {n_high_after} (mean/spread)')

    # ---- Full-feature heatmap incl. categorical + missingness ------
    # Mirror the model's input layout: continuous mean/spread features
    # (post-clip, NaN -> 0) + AIDS/HEM/METS (binary) + 2 Adm_type
    # dummies (medical=0 is the reference; scheduled=1, unscheduled=2)
    # + M_PaO2 / M_bilirubin (computed from raw NaN before fill).
    print('\n=== Full feature correlation incl. categorical + M_* ===')
    df_full = df_ms.copy()
    df_full['AIDS'] = df['AIDS'].astype(float).values
    df_full['HEM']  = df['HEM'].astype(float).values
    df_full['METS'] = df['METS'].astype(float).values
    df_full['Adm_type_scheduled']   = (df['Adm_type'] == 1).astype(float).values
    df_full['Adm_type_unscheduled'] = (df['Adm_type'] == 2).astype(float).values
    df_full['M_PaO2FiO2_vent_min']  = df['PaO2FiO2_vent_min'].isna().astype(float).values
    df_full['M_bilirubin_mean']     = df_ms_raw['bilirubin_mean'].isna().astype(float).values

    cols_full = (cols_ms
                 + ['AIDS', 'HEM', 'METS',
                    'Adm_type_scheduled', 'Adm_type_unscheduled',
                    'M_PaO2FiO2_vent_min', 'M_bilirubin_mean'])
    corr_full = plot_corr_heatmap(
        df_full, cols_full,
        out_name='correlation_full.png',
        title='Pearson correlation: mean/spread + categorical + M_* (NaN -> 0)')
    full_path = os.path.join(OUT, 'clip99_correlation_full.csv')
    corr_full.to_csv(full_path)
    print(f'wrote {full_path}: {corr_full.shape}')

    cm_full = corr_full.values.copy()
    np.fill_diagonal(cm_full, 0.0)
    pairs_full = sorted(
        ((cols_full[i], cols_full[j], cm_full[i, j])
         for i in range(len(cols_full))
         for j in range(i + 1, len(cols_full))),
        key=lambda t: abs(t[2]), reverse=True)
    print('\ntop 15 |corr| pairs (full incl. categorical):')
    for a, b, v in pairs_full[:15]:
        print(f'  {a:24s}  {b:24s}  {v:+.3f}')


if __name__ == '__main__':
    main()
