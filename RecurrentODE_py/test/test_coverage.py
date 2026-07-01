"""Parallel coverage test across RecurrentODE_py estimation modules.

Runs ``--rep`` replications for a chosen ``--module`` / ``--setting`` /
``--knots`` combination using ``multiprocessing.Pool`` and reports bias,
empirical SE, mean sandwich SE and the 95% Wald-CI coverage of the true
``beta = (1, 1, 1)``.

For each successful replication the full per-seed ``_se`` file (containing
``est_r`` and ``se_all`` / ``se_beta`` plus spline metadata like ``knots``
and ``k``) is persisted to ``--save_to/<slug>/seed<seed>.npz`` so
downstream code can build confidence bands for the functional
(spline-coefficient) parameters as well as the β̂ Wald intervals.

Example
-------
    python3 -m RecurrentODE_py.test.test_coverage \
        --module aft --setting 2 --knots quantile --N 1000 --rep 20 --workers 10

    python3 -m RecurrentODE_py.test.test_coverage \
        --module random_effect.ltm --setting 1 --knots K4 --N 1000 --rep 100 --workers 10
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
import time as _time
from importlib import import_module

import numpy as np


# Per-module adapters: how to invoke `main`, how to locate the SE file,
# and how to align beta / SE with the true beta vector.
#
# `beta_slice` and `se_slice` are Python slices used to pull the beta
# estimate out of `est_r` and the matching SE out of `se_all`. For LTM
# the first beta is fixed at 1 and is not in `se_all`, so we drop it:
# `beta_slice = slice(1, 3)` selects (beta2, beta3) and the truth
# vector `true_beta` is likewise trimmed with `beta_slice`.
#
# `file_fmt` is a format string OR a callable (N, seed, setting, knots) -> path.
# It is interpreted relative to ``<root>/res/``.
MODULE_SPECS = {
    'cox': {
        'import': 'RecurrentODE_py.cox.main',
        'knots_arg': False,
        'file_fmt':
            'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.cox': {
        # Same fit/inference as the plain cox module; generator swaps in
        # covariate-dependent censoring drawn from a Cox survival model.
        'import': 'RecurrentODE_py.inform_censor.cox.main',
        'knots_arg': False,
        'file_fmt':
            'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.aft': {
        # Same fit/inference as the plain aft module; generator uses
        # setting-2 AFT intensity + covariate-dependent Cox survival censoring.
        'import': 'RecurrentODE_py.inform_censor.aft.main',
        'knots_arg': True,
        'file_fmt':
            'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.cox_aftc': {
        # Cox recurrent events + AFT-type survival censoring.
        'import': 'RecurrentODE_py.inform_censor.cox_aftc.main',
        'knots_arg': False,
        'file_fmt':
            'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.aft_aftc': {
        # AFT recurrent events + AFT-type survival censoring.
        'import': 'RecurrentODE_py.inform_censor.aft_aftc.main',
        'knots_arg': True,
        'file_fmt':
            'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.cox_unif': {
        # Cox recurrent events + Uniform(a(x), b(x)) censoring.
        'import': 'RecurrentODE_py.inform_censor.cox_unif.main',
        'knots_arg': False,
        'file_fmt':
            'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'inform_censor.aft_unif': {
        # AFT recurrent events + Uniform(a(x), b(x)) censoring.
        'import': 'RecurrentODE_py.inform_censor.aft_unif.main',
        'knots_arg': True,
        'file_fmt':
            'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'aft': {
        'import': 'RecurrentODE_py.aft.main',
        'knots_arg': True,
        'file_fmt':
            'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'ltm': {
        'import': 'RecurrentODE_py.ltm.main',
        'knots_arg': True,
        'file_fmt':
            'res_ltm_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        # LTM fixes beta_1 = 1 for identifiability, so est_r = [1, b2, b3, ...]
        # and se_all = [se(b2), se(b3), se(theta/alpha)...] (no slot for b1).
        'beta_slice': slice(1, 3),
        'se_slice':   slice(0, 2),
    },
    'npmle': {
        'import': 'RecurrentODE_py.npmle.main',
        'knots_arg': True,
        'file_fmt':
            'res_Gtransform_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'random_effect.cox': {
        'import': 'RecurrentODE_py.random_effect.cox.main',
        'knots_arg': False,
        'file_fmt':
            'res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
        'se_key': 'se_beta',
    },
    'random_effect.aft_rec': {
        'import': 'RecurrentODE_py.random_effect.aft_rec.main',
        'knots_arg': True,
        'file_fmt':
            'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        'beta_slice': slice(0, 3),
        'se_slice':   slice(0, 3),
    },
    'random_effect.ltm': {
        'import': 'RecurrentODE_py.random_effect.ltm.main',
        'knots_arg': True,
        # Result path depends on setting: setting 1 writes to
        # res/cox_rec_rd/res_cox_N...; setting 2 writes to res/aft_rd/res_aft_N...
        # The SE file is "<...>_fish.npz" (not "_se.npz").
        'file_fmt': (lambda N, seed, setting, knots: (
            'cox_rec_rd/'
            f'res_cox_N{N}_seed{seed}_setting{setting}_knots{knots}_fish.npz'
        ) if setting == 1 else (
            'aft_rd/'
            f'res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_fish.npz'
        )),
        # RE-LTM also fixes beta_1 = 1 for identifiability.
        'beta_slice': slice(1, 3),
        'se_slice':   slice(0, 2),
    },
}


def _resolve_se_path(spec, N, seed, setting, knots, root):
    fmt = spec['file_fmt']
    if callable(fmt):
        rel = fmt(N, seed, setting, knots)
    else:
        rel = fmt.format(N=N, seed=seed, setting=setting, knots=knots)
    return os.path.join(root, 'res', rel)


def _call_main(module_name, N, seed, setting, knots, root):
    spec = MODULE_SPECS[module_name]
    mod = import_module(spec['import'])
    if spec['knots_arg']:
        return mod.main(N, seed, setting, knots, True, root)
    # (N, seed, data_setting, ci_flag, root) positionally — the ci kwarg
    # is named `calculate_ci` in cox/main.py and `ci` in
    # random_effect.cox/main.py, so avoid kwarg altogether.
    return mod.main(N, seed, setting, True, root)


def _worker(args):
    module_name, N, seed, setting, knots, root, save_to = args
    spec = MODULE_SPECS[module_name]
    t0 = _time.time()
    try:
        _call_main(module_name, N, seed, setting, knots, root)
        se_path = _resolve_se_path(spec, N, seed, setting, knots, root)
        if not os.path.isfile(se_path):
            return seed, None, None, _time.time() - t0, f'no SE file: {se_path}'
        se_mat = np.load(se_path)
        se_key = spec.get('se_key', 'se_all')
        beta_full = np.asarray(se_mat['est_r']).ravel()
        se_full = np.asarray(se_mat[se_key]).ravel()
        beta = beta_full[spec['beta_slice']]
        se = se_full[spec['se_slice']]
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            shutil.copyfile(se_path, os.path.join(save_to, f'seed{seed}.npz'))
        return seed, beta, se, _time.time() - t0, None
    except Exception as exc:
        return seed, None, None, _time.time() - t0, repr(exc)


def _default_slug(module_name, setting, knots):
    safe = module_name.replace('.', '_')
    if knots:
        return f'{safe}_setting{setting}_knots{knots}'
    return f'{safe}_setting{setting}'


def _aggregate_from_save_to(save_to, spec):
    """Scan save_to/seed*.npz and compute (betas, ses) over every persisted
    seed. Used for _summary.npz so that appended runs aggregate over the
    full accumulated set, not just the current batch."""
    import glob, re
    betas, ses, seeds = [], [], []
    se_key = spec.get('se_key', 'se_all')
    for p in sorted(glob.glob(os.path.join(save_to, 'seed*.npz'))):
        m = re.search(r'seed(\d+)\.npz$', p)
        if not m:
            continue
        try:
            mat = np.load(p)
            beta_full = np.asarray(mat['est_r']).ravel()
            se_full = np.asarray(mat[se_key]).ravel()
            betas.append(beta_full[spec['beta_slice']])
            ses.append(se_full[spec['se_slice']])
            seeds.append(int(m.group(1)))
        except Exception:
            continue
    return np.asarray(betas), np.asarray(ses), seeds


def run(module_name, N, rep, setting, knots, workers,
        root=None, save_to=None, seed_start=None, seed_end=None,
        skip_existing=True):
    """Run seeds in the half-open range [seed_start, seed_end] inclusive.

    Default behaviour (``seed_start=None``) is the legacy ``1..rep`` range.
    When extending a prior run, pass ``seed_start=prev_rep+1``,
    ``seed_end=prev_rep+new_rep`` so previously persisted seeds are skipped
    (idempotent) and ``_summary.npz`` is re-aggregated over every seed file
    present in ``save_to``.
    """
    if module_name not in MODULE_SPECS:
        raise ValueError(f'unknown module {module_name}')
    spec = MODULE_SPECS[module_name]
    if root is None:
        root = tempfile.mkdtemp(prefix=f'{module_name}_test_')
    if save_to is None:
        repo_test = os.path.dirname(os.path.abspath(__file__))
        save_to = os.path.join(
            repo_test, 'results', _default_slug(module_name, setting, knots),
        )
    os.makedirs(save_to, exist_ok=True)
    if seed_start is None:
        seed_start = 1
    if seed_end is None:
        seed_end = seed_start + rep - 1
    seeds_wanted = list(range(seed_start, seed_end + 1))
    if skip_existing:
        seeds = [s for s in seeds_wanted
                 if not os.path.isfile(os.path.join(save_to, f'seed{s}.npz'))]
    else:
        seeds = list(seeds_wanted)
    print(f'module={module_name} setting={setting} knots={knots!r} '
          f'N={N} workers={workers}')
    print(f'seeds   : [{seed_start}..{seed_end}] '
          f'({len(seeds)} to run, {len(seeds_wanted) - len(seeds)} already cached)')
    print(f'workdir : {root}')
    print(f'save_to : {save_to}')

    jobs = [(module_name, N, s, setting, knots, root, save_to) for s in seeds]

    t0 = _time.time()
    betas, ses, wall, failures = [], [], [], []
    if jobs:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            completed = 0
            for seed, beta, se, rt, err in pool.imap_unordered(_worker, jobs):
                completed += 1
                if err is not None:
                    failures.append((seed, err))
                    print(f'  [{module_name}] seed {seed} FAILED: {err}',
                          flush=True)
                else:
                    betas.append(beta); ses.append(se); wall.append(rt)
                    print(f'  [{module_name}] {completed:>3}/{len(jobs)} '
                          f'seed={seed:<4} beta={beta.round(3)} '
                          f'se={se.round(3)} t={rt:.1f}s', flush=True)
    total = _time.time() - t0

    # Aggregate over every persisted seed in save_to (includes prior batches).
    all_betas, all_ses, all_seeds = _aggregate_from_save_to(save_to, spec)
    if all_betas.size == 0:
        print(f'\nno persisted seeds in {save_to}; total wall {total:.1f}s')
        return None
    true_beta = np.ones(all_betas.shape[1])
    mean_est = all_betas.mean(0)
    bias = mean_est - true_beta
    emp_se = all_betas.std(0, ddof=1)
    mean_se = all_ses.mean(0)
    lower = all_betas - 1.96 * all_ses
    upper = all_betas + 1.96 * all_ses
    coverage = ((lower < true_beta) & (upper > true_beta)).mean(0)
    rep_total = all_betas.shape[0]

    print(f'\n=== {module_name} / setting {setting} / '
          f'knots={knots!r} / N={N} / rep_total={rep_total} '
          f'(new={len(betas)}) ===')
    print(f'  mean_est   = {mean_est.round(4)}')
    print(f'  bias       = {bias.round(4)}')
    print(f'  emp_se     = {emp_se.round(4)}')
    print(f'  mean_se    = {mean_se.round(4)}')
    print(f'  coverage95 = {coverage.round(3)}')
    if wall:
        print(f'  avg_fit    = {np.mean(wall):.2f}s  '
              f'(batch wall {total:.1f}s)')
    print(f'  failures   = {len(failures)} (this batch)')
    print(f'  per-seed files in {save_to} ({rep_total} total)')

    summary_path = os.path.join(save_to, '_summary.npz')
    # Preserve total_wall across batches: add this batch to whatever was
    # recorded before.
    prev_total_wall = 0.0
    prev_failures = 0
    if os.path.isfile(summary_path):
        try:
            prev = np.load(summary_path, allow_pickle=True)
            prev_total_wall = float(np.atleast_1d(prev['total_wall'])[0])
            prev_failures = int(np.atleast_1d(prev['n_failures'])[0])
        except Exception:
            pass
    avg_fit_val = float(np.mean(wall)) if wall else 0.0
    np.savez_compressed(
        summary_path,
        module=module_name, setting=setting, knots=knots or '',
        N=N, rep=rep_total, workers=workers,
        mean_est=mean_est, bias=bias,
        emp_se=emp_se, mean_se=mean_se, coverage=coverage,
        avg_fit=avg_fit_val, total_wall=prev_total_wall + total,
        n_failures=prev_failures + len(failures),
        seed_start=seed_start, seed_end=seed_end,
    )
    print(f'  summary    = {summary_path}')

    return {
        'module': module_name, 'setting': setting, 'knots': knots,
        'N': N, 'rep': rep_total, 'workers': workers,
        'root': root, 'save_to': save_to,
        'mean_est': mean_est, 'bias': bias, 'emp_se': emp_se,
        'mean_se': mean_se, 'coverage': coverage,
        'avg_fit': avg_fit_val, 'total_wall': prev_total_wall + total,
        'failures': failures,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True,
                        choices=list(MODULE_SPECS.keys()))
    parser.add_argument('--setting', type=int, required=True)
    parser.add_argument('--knots', type=str, default='')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--rep', type=int, default=20)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--root', type=str, default=None,
                        help='workdir for simulated data / fits (default: tempfile)')
    parser.add_argument('--save_to', type=str, default=None,
                        help='dir to persist per-seed est+SE .npz '
                             '(default: RecurrentODE_py/test/results/<slug>/)')
    parser.add_argument('--seed_start', type=int, default=None)
    parser.add_argument('--seed_end', type=int, default=None)
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()
    run(args.module, args.N, args.rep, args.setting, args.knots,
        args.workers, args.root, args.save_to,
        seed_start=args.seed_start, seed_end=args.seed_end,
        skip_existing=not args.no_skip_existing)
