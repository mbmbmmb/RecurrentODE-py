"""Coverage test for the plain ``cox/`` module.

Runs ``rep`` replications of ``cox.main.main`` for each requested
``data_setting`` and reports, per setting:

* the empirical bias of ``beta``,
* the empirical standard deviation of ``beta`` across replications,
* the mean sandwich standard error from ``cox.inference``,
* the empirical coverage of the 95% Wald interval for the true
  ``beta = (1, 1, 1)``.

The intensity functions used by ``generator_rec`` cover:

* setting 1 — Cox proportional hazards (correctly specified here),
* setting 2 — AFT-type recurrent event,
* setting 3 — Box-Cox transformation (``rho1 = 0.5``),
* setting 4 — general linear transformation model.

All output files are written under a per-run temporary directory so the
module's own ``data/`` and ``res/`` are left untouched.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time as _time
from dataclasses import dataclass, field

import numpy as np

from RecurrentODE_py.cox.main import main as run_main


@dataclass
class SettingResult:
    setting: int
    N: int
    rep: int
    mean_est: np.ndarray
    bias: np.ndarray
    empirical_se: np.ndarray
    mean_model_se: np.ndarray
    coverage: np.ndarray
    avg_fit_time: float
    failures: int = 0
    seeds_failed: list[int] = field(default_factory=list)


def run_one_setting(N, rep, data_setting, root):
    p = 3
    est_beta = np.full((rep, p), np.nan)
    est_se = np.full((rep, p), np.nan)
    wall = np.full(rep, np.nan)
    failures = 0
    seeds_failed = []
    total_start = _time.time()
    for r in range(rep):
        seed = r + 1
        try:
            est, rt = run_main(
                N, seed, data_setting, calculate_ci=True, root=root,
            )
            se_mat = np.load(os.path.join(
                root, 'res',
                f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
            ))
            est_beta[r] = est[:p]
            est_se[r] = se_mat['se_all'].ravel()[:p]
            wall[r] = rt
        except Exception as exc:
            failures += 1
            seeds_failed.append(seed)
            print(f'  [setting {data_setting}] seed {seed} failed: {exc}')
            continue
        if (r + 1) % 10 == 0 or (r + 1) == rep:
            print(
                f'  [setting {data_setting}] {r + 1:>3}/{rep}  '
                f'beta={est_beta[r].round(3)}  '
                f'se={est_se[r].round(3)}  '
                f'elapsed={_time.time() - total_start:.1f}s',
                flush=True,
            )

    mask = ~np.isnan(est_beta[:, 0])
    est_beta = est_beta[mask]
    est_se = est_se[mask]
    true_beta = np.ones(p)

    mean_est = est_beta.mean(0)
    bias = mean_est - true_beta
    empirical_se = est_beta.std(0, ddof=1)
    mean_model_se = est_se.mean(0)
    upper = est_beta + 1.96 * est_se
    lower = est_beta - 1.96 * est_se
    coverage = ((lower < true_beta) & (upper > true_beta)).mean(0)
    avg_fit = float(np.nanmean(wall))

    return SettingResult(
        setting=data_setting, N=N, rep=rep,
        mean_est=mean_est, bias=bias,
        empirical_se=empirical_se, mean_model_se=mean_model_se,
        coverage=coverage, avg_fit_time=avg_fit,
        failures=failures, seeds_failed=seeds_failed,
    )


def run(N, rep, settings, root=None):
    if root is None:
        root = tempfile.mkdtemp(prefix='cox_test_')
    print(f'workdir: {root}')
    results = []
    for s in settings:
        print(f'\n=== setting {s} | N={N} rep={rep} ===')
        results.append(run_one_setting(N, rep, s, root))
    print_summary(results)
    return results, root


def print_summary(results):
    print('\n=== cox module coverage summary ===')
    for r in results:
        print(
            f'setting {r.setting}: '
            f'mean_est={r.mean_est.round(4)}  '
            f'bias={r.bias.round(4)}  '
            f'emp_se={r.empirical_se.round(4)}  '
            f'mean_se={r.mean_model_se.round(4)}  '
            f'CP={r.coverage.round(3)}  '
            f'avg_fit={r.avg_fit_time:.2f}s  '
            f'fails={r.failures}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--rep', type=int, default=100)
    parser.add_argument('--settings', type=int, nargs='+',
                        default=[1, 2, 3, 4])
    parser.add_argument('--root', type=str, default=None,
                        help='workdir (default: tempfile.mkdtemp)')
    args = parser.parse_args()
    run(args.N, args.rep, args.settings, args.root)
