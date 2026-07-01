"""Plot the distribution of event and censoring times.

Reads cohort_long.npz and produces two histograms:
  - event_time.png   : time of each readmission (delta == 1)
  - censor_time.png  : end-of-follow-up time per patient (delta == 0)
Both use days since first admission.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ = os.path.join(ROOT, 'merged_data/cohort_long.npz')
OUT = os.path.join(ROOT, 'doc/plots/clip99/outcome')


def stats_legend(s):
    return [
        plt.Line2D([], [], color='none', label=f'n        = {len(s)}'),
        plt.Line2D([], [], color='none', label=f'min      = {s.min():.2f}'),
        plt.Line2D([], [], color='none', label=f'max      = {s.max():.2f}'),
        plt.Line2D([], [], color='none', label=f'mean     = {s.mean():.2f}'),
        plt.Line2D([], [], color='none', label=f'median   = {np.median(s):.2f}'),
        plt.Line2D([], [], color='none', label=f'p99      = {np.quantile(s, 0.99):.2f}'),
    ]


def hist(values, title, xlabel, fname, color):
    s = pd.Series(values).astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(s, bins=80, color=color, alpha=0.85)
    ax.axvline(np.median(s), color='black', ls='--', lw=0.9,
               label=f'median = {np.median(s):.1f}')
    ax.legend(handles=stats_legend(s), loc='upper right',
              fontsize=9, framealpha=0.9, handlelength=0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    npz = np.load(NPZ, allow_pickle=True)
    time, delta, ids = npz['time'], npz['delta'], npz['id']
    print(f'cohort_long.npz : rows={len(time)}  patients={len(np.unique(ids))}')
    print(f'  delta=1 rows  : {int((delta == 1).sum())}')
    print(f'  delta=0 rows  : {int((delta == 0).sum())}')

    evt = time[delta == 1]
    cen = time[delta == 0]

    # Linear-axis hist
    hist(evt, f'Event times (delta=1)   n={len(evt)}',
         'days since first admission', 'event_time.png', '#c0392b')
    hist(cen, f'Censoring times (delta=0)   n={len(cen)}',
         'days since first admission', 'censor_time.png', '#2e86ab')

    # Log-scale variants (the right tails extend to ~10 yr)
    for vals, name, title, color in [
        (evt, 'event_time_log', 'Event times (log10 days)', '#c0392b'),
        (cen, 'censor_time_log', 'Censoring times (log10 days)', '#2e86ab'),
    ]:
        s = np.log10(np.maximum(vals, 1e-6))
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.hist(s, bins=80, color=color, alpha=0.85)
        ax.axvline(np.median(s), color='black', ls='--', lw=0.9)
        ax.legend(handles=stats_legend(pd.Series(vals)),
                  loc='upper right', fontsize=9,
                  framealpha=0.9, handlelength=0)
        ax.set_title(title + f'   n={len(vals)}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('log10(days)')
        ax.set_ylabel('count')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=110)
        plt.close(fig)

    # Quantile report
    print('\n=== Event times (days, delta=1) ===')
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f'  q{int(q*100):02d} = {np.quantile(evt, q):8.2f}')
    print('\n=== Censoring times (days, delta=0) ===')
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f'  q{int(q*100):02d} = {np.quantile(cen, q):8.2f}')

    print(f'\nwrote 4 PNGs to {OUT}')


if __name__ == '__main__':
    main()
