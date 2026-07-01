"""RE-cox functional-parameter (baseline hazard) band coverage.

``random_effect.cox`` only stores ``se_beta`` (closed-form β sandwich)
in its per-seed file.  The spline SEs require
``random_effect/cox/inference.py`` (resampling B=50).  This driver:

1. For each seed in parallel: regenerates data (deterministic on seed),
   calls ``main(..., ci=True)`` to get ``est_r``, then calls
   ``inference(...)`` to get ``se_all`` (length ``p + q``).
2. Builds the pointwise Wald band
       exp(B(u) @ (theta_hat ± 1.96 se_theta))
   on a grid and checks coverage of the true baseline hazard
   ``lambda_0(t) = t^2 + 1`` (setting 1).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
import time

import numpy as np

from RecurrentODE_py.common import spcol
from RecurrentODE_py.random_effect.cox.main import main as cox_rd_main
from RecurrentODE_py.random_effect.cox.inference import inference as cox_rd_inf


def _one_seed(args):
    N, seed, data_setting = args
    root = tempfile.mkdtemp(prefix=f'reCoxInf_seed{seed}_')
    try:
        t0 = time.time()
        cox_rd_main(N, seed, data_setting, True, root=root)
        cox_rd_inf(N, seed, data_setting, root=root)
        res = np.load(os.path.join(
            root, 'res',
            f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz'))
        inf = np.load(os.path.join(
            root, 'res',
            f'res_cox_N{N}_seed{seed}_setting{data_setting}_inference.npz'))
        est_r = res['est_r'].ravel()
        p = int(res['p'].ravel()[0])
        k = int(res['k'].ravel()[0])
        knots = res['knots'].ravel()
        se_all = inf['se_all'].ravel()
        theta = est_r[p:]
        se_theta = se_all[p:]
        return dict(seed=seed, ok=True, theta=theta, se_theta=se_theta,
                    knots=knots, k=k, p=p, t=time.time() - t0)
    except Exception as exc:
        return dict(seed=seed, ok=False, err=repr(exc), t=time.time() - t0)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--rep', type=int, default=100)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--data_setting', type=int, default=1)
    a = ap.parse_args()

    jobs = [(a.N, s, a.data_setting) for s in range(1, a.rep + 1)]
    grid = np.linspace(0.1, 1.6, 30)       # baseline hazard support
    true_curve = grid ** 2 + 1.0

    est_all, up_all, lo_all = [], [], []
    t0 = time.time()
    ok = 0
    print(f'Launching {a.rep} seeds on {a.workers} workers ...')
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one_seed, jobs), 1):
            if not r['ok']:
                print(f'  [{i:3d}/{a.rep}] seed={r["seed"]:4d}  FAILED  {r.get("err")}')
                continue
            ok += 1
            B = spcol(r['knots'], r['k'], grid)
            est = np.exp(B @ r['theta'])
            up  = np.exp(B @ (r['theta'] + 1.96 * r['se_theta']))
            lo  = np.exp(B @ (r['theta'] - 1.96 * r['se_theta']))
            est_all.append(est); up_all.append(up); lo_all.append(lo)
            if i % 10 == 0:
                elapsed = time.time() - t0
                print(f'  [{i:3d}/{a.rep}] seed={r["seed"]:4d}  t={r["t"]:.1f}s  elapsed={elapsed:.0f}s')

    est_all = np.asarray(est_all); up_all = np.asarray(up_all)
    lo_all = np.asarray(lo_all)
    inside = (lo_all <= true_curve[None, :]) & (true_curve[None, :] <= up_all)
    pw = inside.mean(axis=0)
    sim = inside.all(axis=1).mean()

    print()
    print(f'=== re_cox / setting {a.data_setting} (baseline alpha(t)=t^2+1)   '
          f'n={ok}/{a.rep} ===')
    print(f'  pointwise coverage: mean={pw.mean():.3f}  '
          f'min={pw.min():.3f}  max={pw.max():.3f}')
    print(f'  simultaneous coverage:              {sim:.3f}')
    print(f'  total wall {time.time() - t0:.0f}s')

    # persist
    out_dir = os.path.join(
        os.path.dirname(__file__), 'results',
        f'random_effect_cox_setting{a.data_setting}_band',
    )
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, '_summary.npz'),
        grid=grid, true_curve=true_curve,
        est=est_all, up=up_all, lo=lo_all,
        pw=pw, sim=sim, n=ok,
    )
    print(f'  persisted -> {out_dir}/_summary.npz')


if __name__ == '__main__':
    main()
