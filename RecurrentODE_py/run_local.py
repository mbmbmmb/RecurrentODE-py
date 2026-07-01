"""Local parallel driver: run every config in ``job_submission/configs.json``
for a chosen number of replications, with a fixed pool of workers.

Layout (per config slug):
  RecurrentODE_py/results/<slug>/data/       generator_rec outputs
  RecurrentODE_py/results/<slug>/res/        per-seed est + SE .npz
  RecurrentODE_py/results/<slug>/summary/    seed{n}.npz copies + _summary.npz

The <slug> encodes module + setting + (optional) knots + N so that multi-N
configs (e.g. npmle with N=2000 and 4000) get distinct folders.

Each (config, N) runs as a single ``test_coverage.run`` call, which itself
uses ``multiprocessing.Pool`` with ``--workers`` seeds in flight.  We run
the configs sequentially so the pool size caps parallelism at --workers.

Usage
-----
    # Smoke test: 2 reps per config, all 7 configs
    python3 -m RecurrentODE_py.run_local --rep 2 --workers 10 --only cox,aft

    # Full 100-rep sweep (matches MATLAB summary.m sample size)
    python3 -m RecurrentODE_py.run_local --rep 100 --workers 10

    # Subset
    python3 -m RecurrentODE_py.run_local --rep 100 --only cox,aft,ltm

Idempotent: seeds whose ``summary/seed{n}.npz`` already exist are NOT rerun
because ``test_coverage._worker`` checks for the SE file on disk first --
so rerunning the same command only recomputes missing seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .test.test_coverage import run as run_coverage, MODULE_SPECS


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'job_submission', 'configs.json')
RESULTS_ROOT = os.path.join(HERE, 'results')


def _load_configs():
    with open(CONFIG_PATH) as f:
        raw = json.load(f)
    defaults = raw.get('defaults', {})
    configs = raw.get('configs', {})
    merged = {}
    for name, c in configs.items():
        m = dict(defaults)
        m.update(c)
        m.setdefault('knots', '')
        m.setdefault('n_values', [1000])
        merged[name] = m
    return merged


def _slug(name, setting, knots, N):
    safe = name.replace('.', '_')
    base = f'{safe}_setting{setting}'
    if knots:
        base += f'_knots{knots}'
    base += f'_N{N}'
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=100,
                    help='seeds per (config, N); ignored if both '
                         '--seed_start and --seed_end are given')
    ap.add_argument('--seed_start', type=int, default=None,
                    help='first seed (inclusive); default 1')
    ap.add_argument('--seed_end', type=int, default=None,
                    help='last seed (inclusive); default seed_start+rep-1')
    ap.add_argument('--workers', type=int, default=10,
                    help='parallel pool size (n_cpus - 2 = 10 on this box)')
    ap.add_argument('--only', type=str, default='',
                    help='comma-separated subset of config names to run')
    ap.add_argument('--skip', type=str, default='',
                    help='comma-separated config names to skip')
    ap.add_argument('--results_root', type=str, default=RESULTS_ROOT)
    args = ap.parse_args()

    configs = _load_configs()
    only = {s for s in args.only.split(',') if s}
    skip = {s for s in args.skip.split(',') if s}
    names = [n for n in configs if (not only or n in only) and n not in skip]
    if only:
        missing = only - set(configs)
        if missing:
            print(f'[run_local] unknown config(s): {sorted(missing)}',
                  file=sys.stderr)
            return 1

    os.makedirs(args.results_root, exist_ok=True)
    log_path = os.path.join(args.results_root, '_runlog.csv')
    new_log = not os.path.isfile(log_path)
    log = open(log_path, 'a', buffering=1)
    if new_log:
        log.write('ts,name,module,setting,knots,N,rep,workers,'
                  'ok,failures,avg_fit_s,total_wall_s,slug\n')

    t_all = time.time()
    print(f'[run_local] configs={names} rep={args.rep} workers={args.workers}')
    print(f'[run_local] results_root={args.results_root}')
    print()

    for name in names:
        c = configs[name]
        module = c['module']
        if module not in MODULE_SPECS:
            print(f'[run_local] skip {name}: module={module} not in MODULE_SPECS')
            continue
        knots = c.get('knots', '') or ''
        setting = int(c['data_setting'])
        for N in c['n_values']:
            N = int(N)
            slug = _slug(name, setting, knots, N)
            root = os.path.join(args.results_root, slug)
            save_to = os.path.join(root, 'summary')
            print(f'=== [{name}] module={module} setting={setting} '
                  f'knots={knots!r} N={N}  slug={slug}')
            t0 = time.time()
            try:
                r = run_coverage(
                    module, N, args.rep, setting, knots,
                    workers=args.workers, root=root, save_to=save_to,
                    seed_start=args.seed_start, seed_end=args.seed_end,
                )
                ok = r is not None
                failures = len(r['failures']) if r else args.rep
                avg = r['avg_fit'] if r else 0.0
                tot = r['total_wall'] if r else (time.time() - t0)
            except Exception as exc:
                print(f'[run_local] {name}/N={N} EXCEPTION: {exc!r}',
                      file=sys.stderr)
                ok, failures, avg, tot = False, args.rep, 0.0, time.time() - t0

            log.write(
                f'{int(time.time())},{name},{module},{setting},{knots},'
                f'{N},{args.rep},{args.workers},{int(ok)},{failures},'
                f'{avg:.3f},{tot:.1f},{slug}\n'
            )
            print(f'    -> ok={ok}  fail={failures}  wall={tot:.1f}s\n')

    log.close()
    print(f'[run_local] DONE  total wall {time.time() - t_all:.0f}s')
    print(f'[run_local] log  : {log_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
