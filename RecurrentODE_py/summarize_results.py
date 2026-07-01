"""Aggregate all ``_summary.npz`` files under ``results/`` into a single
table: one row per (config, N), columns mean_est / bias / emp_se / mean_se /
coverage / avg_fit / total_wall / n_failures.

Writes:
  RecurrentODE_py/results/_summary.csv
  RecurrentODE_py/results/_summary.md

Run:
  python3 -m RecurrentODE_py.summarize_results
"""
from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')


def _fmt_vec(v):
    return '[' + ' '.join(f'{x:.4f}' for x in np.atleast_1d(v)) + ']'


def main():
    rows = []
    for name in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, name, 'summary', '_summary.npz')
        if not os.path.isfile(p):
            continue
        s = np.load(p, allow_pickle=True)
        row = dict(
            slug=name,
            module=str(s['module']),
            setting=int(np.atleast_1d(s['setting'])[0]),
            knots=str(s['knots']),
            N=int(np.atleast_1d(s['N'])[0]),
            rep=int(np.atleast_1d(s['rep'])[0]),
            mean_est=np.atleast_1d(s['mean_est']),
            bias=np.atleast_1d(s['bias']),
            emp_se=np.atleast_1d(s['emp_se']),
            mean_se=np.atleast_1d(s['mean_se']),
            coverage=np.atleast_1d(s['coverage']),
            avg_fit=float(np.atleast_1d(s['avg_fit'])[0]),
            total_wall=float(np.atleast_1d(s['total_wall'])[0]),
            n_failures=int(np.atleast_1d(s['n_failures'])[0]),
        )
        rows.append(row)

    csv_path = os.path.join(RESULTS, '_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slug', 'module', 'setting', 'knots', 'N', 'rep',
                    'mean_est', 'bias', 'emp_se', 'mean_se', 'coverage',
                    'avg_fit_s', 'total_wall_s', 'n_failures'])
        for r in rows:
            w.writerow([
                r['slug'], r['module'], r['setting'], r['knots'], r['N'],
                r['rep'],
                _fmt_vec(r['mean_est']), _fmt_vec(r['bias']),
                _fmt_vec(r['emp_se']), _fmt_vec(r['mean_se']),
                _fmt_vec(r['coverage']),
                f'{r["avg_fit"]:.3f}', f'{r["total_wall"]:.1f}',
                r['n_failures'],
            ])
    print(f'wrote {csv_path}  ({len(rows)} rows)')

    md_path = os.path.join(RESULTS, '_summary.md')
    with open(md_path, 'w') as f:
        f.write('# 100-rep simulation summary (local run)\n\n')
        f.write('Each row = one config from `job_submission/configs.json` '
                'run with `rep=100` and `workers=10` via '
                '`RecurrentODE_py.run_local`. Layout per slug: '
                '`results/<slug>/{data,res,summary}/`.\n\n')
        for r in rows:
            f.write(f'## {r["slug"]}\n\n')
            f.write(f'- module `{r["module"]}`, setting {r["setting"]}, '
                    f'knots `{r["knots"] or "-"}`, N={r["N"]}, '
                    f'rep={r["rep"]}, failures={r["n_failures"]}\n')
            f.write(f'- mean_est   {_fmt_vec(r["mean_est"])}\n')
            f.write(f'- bias       {_fmt_vec(r["bias"])}\n')
            f.write(f'- emp_se     {_fmt_vec(r["emp_se"])}\n')
            f.write(f'- mean_se    {_fmt_vec(r["mean_se"])}\n')
            f.write(f'- coverage95 {_fmt_vec(r["coverage"])}\n')
            f.write(f'- timing     avg_fit={r["avg_fit"]:.2f}s  '
                    f'total_wall={r["total_wall"]:.1f}s\n\n')
    print(f'wrote {md_path}')

    # Compact table also echoed to stdout.
    print()
    print(f'{"slug":45s} {"cov":15s} {"bias":25s} {"wall":>7s}')
    print('-' * 100)
    for r in rows:
        cov = ' '.join(f'{x:.2f}' for x in r['coverage'][:3])
        bias = ' '.join(f'{x:+.3f}' for x in r['bias'][:3])
        print(f'{r["slug"]:45s} {cov:15s} {bias:25s} {r["total_wall"]:7.1f}s')


if __name__ == '__main__':
    main()
