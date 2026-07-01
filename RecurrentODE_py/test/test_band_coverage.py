"""Functional-parameter coverage (confidence-band) check.

Reads the per-seed ``.npz`` files in ``RecurrentODE_py/test/results/<slug>/``
that were produced by ``test_coverage.py``, reconstructs the pointwise
Wald band

    est_q(u)        = exp( B(u) @ theta_hat )
    est_q_upper(u)  = exp( B(u) @ (theta_hat + 1.96 * se_theta) )
    est_q_lower(u)  = exp( B(u) @ (theta_hat - 1.96 * se_theta) )

(as in each module's ``visual.py`` / ``visual.m``), and reports

- **pointwise coverage**: at each grid point ``u``, the fraction of
  replications whose band contains the true value; averaged across the
  grid (+ min/max along the grid).
- **simultaneous coverage**: fraction of replications whose band covers
  the truth at **every** grid point.

Covers: cox (s1), aft (s2), ltm (s4), npmle (s3), random_effect.aft_rec
(s2), random_effect.ltm (s1).  random_effect.cox stores only ``se_beta``
in its per-seed file, so its functional-parameter coverage is computed
separately by ``test_band_coverage_re_cox.py``.
"""
from __future__ import annotations

import os
import numpy as np

from RecurrentODE_py.common import spcol


RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def _band_single_spline(theta, se_theta, knots, k, grid):
    B = spcol(knots, k, grid)
    est = np.exp(B @ theta)
    up  = np.exp(B @ (theta + 1.96 * se_theta))
    lo  = np.exp(B @ (theta - 1.96 * se_theta))
    return est, up, lo


def _band_ltm(est_r, se_all, spec, grid_t, grid_u, scale_a=1.0, scale_q=1.0):
    """LTM / RE-LTM: two splines (alpha(t), q(u)).

    ``spec`` carries the layout: ``p``, ``q_q``, ``q_0``, ``knots_0``,
    ``knots_q``, ``k0``, ``kq``, ``se_offset`` (``p-1`` when beta_1 is
    fixed at 1 — LTM and RE-LTM — else ``p``).
    """
    p = int(spec['p']); q_q = int(spec['q_q']); q_0 = int(spec['q_0'])
    knots_0 = spec['knots_0']; knots_q = spec['knots_q']
    k0 = int(spec['k0']); kq = int(spec['kq'])
    se_off = int(spec['se_offset'])

    theta = est_r[p:p + q_q]
    alpha = est_r[p + q_q:p + q_q + q_0]
    se_theta = se_all[se_off:se_off + q_q]
    se_alpha = se_all[se_off + q_q:se_off + q_q + q_0]

    Bq = spcol(knots_q, kq, grid_u)
    q, qu, ql = (np.exp(Bq @ theta),
                 np.exp(Bq @ (theta + 1.96 * se_theta)),
                 np.exp(Bq @ (theta - 1.96 * se_theta)))
    B0 = spcol(knots_0, k0, grid_t)
    a, au, al = (np.exp(B0 @ alpha),
                 np.exp(B0 @ (alpha + 1.96 * se_alpha)),
                 np.exp(B0 @ (alpha - 1.96 * se_alpha)))
    return (a * scale_a, au * scale_a, al * scale_a,
            q * scale_q, qu * scale_q, ql * scale_q)


def _coverage(band_est, band_up, band_lo, true_curve):
    """``band_*`` have shape (rep, grid); returns pointwise (mean/min/max)
    and simultaneous coverage."""
    inside = (band_lo <= true_curve[None, :]) & (true_curve[None, :] <= band_up)
    pw = inside.mean(axis=0)
    sim = inside.all(axis=1).mean()
    return pw.mean(), pw.min(), pw.max(), float(sim)


def _load_seed(slug, seed):
    path = os.path.join(RESULTS, slug, f'seed{seed}.npz')
    if not os.path.isfile(path):
        return None
    return np.load(path)


def _iter_seeds(slug):
    d = os.path.join(RESULTS, slug)
    if not os.path.isdir(d):
        return []
    seeds = []
    for name in os.listdir(d):
        if name.startswith('seed') and name.endswith('.npz'):
            try:
                seeds.append(int(name[4:-4]))
            except ValueError:
                pass
    return sorted(seeds)


# ---------- module-specific drivers ----------

def cover_single(slug, grid, truth_fn):
    seeds = _iter_seeds(slug)
    bands_est, bands_up, bands_lo = [], [], []
    for s in seeds:
        f = _load_seed(slug, s)
        if f is None: continue
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        p = int(f['p'].ravel()[0]); k = int(f['k'].ravel()[0])
        knots = f['knots'].ravel()
        theta = est_r[p:]; se = se_all[p:]
        e, u, l = _band_single_spline(theta, se, knots, k, grid)
        bands_est.append(e); bands_up.append(u); bands_lo.append(l)
    be = np.asarray(bands_est); bu = np.asarray(bands_up); bl = np.asarray(bands_lo)
    true_curve = truth_fn(grid)
    m, mn, mx, sim = _coverage(be, bu, bl, true_curve)
    return dict(n=len(bands_est), mean_pw=m, min_pw=mn, max_pw=mx, sim=sim,
                true_curve=true_curve, be=be, bu=bu, bl=bl)


def cover_ltm_like(slug, data_setting, grid_t, grid_u, truth_a, truth_q,
                   scale_a=1.0, scale_q=1.0, se_offset_rule='p-1'):
    """Handles LTM (data_setting=4) and RE-LTM (data_setting=1)."""
    seeds = _iter_seeds(slug)
    Ae, Au, Al, Qe, Qu, Ql = [], [], [], [], [], []
    for s in seeds:
        f = _load_seed(slug, s)
        if f is None: continue
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        q_q = int(f['q_q'].ravel()[0]); q_0 = int(f['q_0'].ravel()[0])
        # plain ltm doesn't persist `p`; infer from est_r layout
        # (est_r carries p betas + q_q theta + q_0 alpha).
        p = (int(f['p'].ravel()[0]) if 'p' in f.files
             else est_r.size - q_q - q_0)
        spec = {
            'p':       p,
            'q_q':     q_q,
            'q_0':     q_0,
            'knots_0': f['knots_0'].ravel(),
            'knots_q': f['knots_q'].ravel(),
            'k0':      int(f['k0'].ravel()[0]),
            'kq':      int(f['kq'].ravel()[0]),
            'se_offset': (p - 1 if se_offset_rule == 'p-1' else p),
        }
        a, au, al, q, qu, ql = _band_ltm(est_r, se_all, spec, grid_t, grid_u,
                                         scale_a=scale_a, scale_q=scale_q)
        Ae.append(a); Au.append(au); Al.append(al)
        Qe.append(q); Qu.append(qu); Ql.append(ql)
    Ae, Au, Al = map(np.asarray, (Ae, Au, Al))
    Qe, Qu, Ql = map(np.asarray, (Qe, Qu, Ql))
    tA = truth_a(grid_t); tQ = truth_q(grid_u)
    aA = _coverage(Ae, Au, Al, tA)
    aQ = _coverage(Qe, Qu, Ql, tQ)
    return dict(
        n=len(Ae),
        alpha=dict(mean_pw=aA[0], min_pw=aA[1], max_pw=aA[2], sim=aA[3],
                   true=tA, est=Ae, up=Au, lo=Al, grid=grid_t),
        q    =dict(mean_pw=aQ[0], min_pw=aQ[1], max_pw=aQ[2], sim=aQ[3],
                   true=tQ, est=Qe, up=Qu, lo=Ql, grid=grid_u),
    )


# ---------- main ----------

def main():
    print('=== Functional-parameter (band) coverage ===\n')

    # cox setting 1: true baseline hazard q(u) = u^2 + 1
    g = np.linspace(0.1, 2.5, 30)
    r = cover_single('cox_setting1', g, lambda u: u ** 2 + 1)
    print(f'cox / setting 1 (baseline q(u)=u^2+1)   n={r["n"]}')
    print(f'  pointwise coverage: mean={r["mean_pw"]:.3f}  '
          f'min={r["min_pw"]:.3f}  max={r["max_pw"]:.3f}')
    print(f'  simultaneous coverage:              {r["sim"]:.3f}')
    print()

    # aft setting 2: true q(u) = 2/(1+u)
    g = np.linspace(0.1, 3.5, 30)
    r = cover_single('aft_setting2_knotsquantile', g, lambda u: 2.0 / (1 + u))
    print(f'aft / setting 2 (q(u)=2/(1+u))          n={r["n"]}')
    print(f'  pointwise coverage: mean={r["mean_pw"]:.3f}  '
          f'min={r["min_pw"]:.3f}  max={r["max_pw"]:.3f}')
    print(f'  simultaneous coverage:              {r["sim"]:.3f}')
    print()

    # npmle setting 3: true q(u) = 0.2/(1+u)
    g = np.linspace(0.05, 2.5, 30)
    r = cover_single('npmle_setting3_knotsequal', g, lambda u: 0.2 / (1 + u))
    print(f'npmle / setting 3 (q(u)=0.2/(1+u))      n={r["n"]}')
    print(f'  pointwise coverage: mean={r["mean_pw"]:.3f}  '
          f'min={r["min_pw"]:.3f}  max={r["max_pw"]:.3f}')
    print(f'  simultaneous coverage:              {r["sim"]:.3f}')
    print()

    # RE-aft setting 2: true q(u) = 2/(1+u)
    g = np.linspace(0.1, 3.5, 30)
    r = cover_single('random_effect_aft_rec_setting2_knotsquantile', g,
                     lambda u: 2.0 / (1 + u))
    print(f're_aft / setting 2 (q(u)=2/(1+u))       n={r["n"]}')
    print(f'  pointwise coverage: mean={r["mean_pw"]:.3f}  '
          f'min={r["min_pw"]:.3f}  max={r["max_pw"]:.3f}')
    print(f'  simultaneous coverage:              {r["sim"]:.3f}')
    print()

    # LTM setting 4: truth_a(t) = t+1,  truth_q(u) = 2/(1+u),  scales (3, 1/3)
    gt = np.linspace(0.1, 2.0, 25); gu = np.linspace(0.1, 2.0, 25)
    r = cover_ltm_like('ltm_setting4_knotsK4', 4, gt, gu,
                       truth_a=lambda t: t + 1,
                       truth_q=lambda u: 2.0 / (1 + u),
                       scale_a=3.0, scale_q=1.0 / 3.0,
                       se_offset_rule='p-1')
    print(f'ltm / setting 4 (alpha(t)=t+1, q(u)=2/(1+u))   n={r["n"]}')
    print(f'  alpha  pointwise: mean={r["alpha"]["mean_pw"]:.3f}  '
          f'min={r["alpha"]["min_pw"]:.3f}  max={r["alpha"]["max_pw"]:.3f}  '
          f'sim={r["alpha"]["sim"]:.3f}')
    print(f'  q      pointwise: mean={r["q"]["mean_pw"]:.3f}  '
          f'min={r["q"]["min_pw"]:.3f}  max={r["q"]["max_pw"]:.3f}  '
          f'sim={r["q"]["sim"]:.3f}')
    print()

    # RE-LTM setting 1: truth_a(t) = t^2 + 1, truth_q(u) = 1
    # RE-LTM visual normalises by alpha(0); we mirror that here (since
    # per-seed alpha(0) is not 1, we divide alpha/q by its value at t=0
    # — log scale in visual is equivalent to dividing alpha by a0 and
    # multiplying q by a0).
    gt = np.linspace(0.1, 2.0, 25); gu = np.linspace(0.1, 10.0, 40)
    # Use se_offset = p-1 (beta_1 fixed).  No fixed scale_a/scale_q —
    # we renormalise per-replication using alpha(0).
    seeds = _iter_seeds('random_effect_ltm_setting1_knotsK4')
    Ae, Au, Al, Qe, Qu, Ql = [], [], [], [], [], []
    for s in seeds:
        f = _load_seed('random_effect_ltm_setting1_knotsK4', s)
        if f is None: continue
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        spec = {
            'p': int(f['p'].ravel()[0]),
            'q_q': int(f['q_q'].ravel()[0]),
            'q_0': int(f['q_0'].ravel()[0]),
            'knots_0': f['knots_0'].ravel(),
            'knots_q': f['knots_q'].ravel(),
            'k0': int(f['k0'].ravel()[0]),
            'kq': int(f['kq'].ravel()[0]),
            'se_offset': int(f['p'].ravel()[0]) - 1,
        }
        # alpha(0) renormalisation:
        a0_grid = np.array([0.0])
        a0, _, _, _, _, _ = _band_ltm(est_r, se_all, spec, a0_grid, a0_grid)
        scale_a = 1.0 / float(a0[0])
        scale_q = float(a0[0])
        a, au, al, q, qu, ql = _band_ltm(est_r, se_all, spec, gt, gu,
                                         scale_a=scale_a, scale_q=scale_q)
        Ae.append(a); Au.append(au); Al.append(al)
        Qe.append(q); Qu.append(qu); Ql.append(ql)
    Ae, Au, Al = map(np.asarray, (Ae, Au, Al))
    Qe, Qu, Ql = map(np.asarray, (Qe, Qu, Ql))
    tA = gt ** 2 + 1.0
    tQ = np.ones_like(gu)
    aA = _coverage(Ae, Au, Al, tA)
    aQ = _coverage(Qe, Qu, Ql, tQ)
    print(f're_ltm / setting 1 (alpha(t)=t^2+1, q(u)=1)   n={len(Ae)}')
    print(f'  alpha  pointwise: mean={aA[0]:.3f}  min={aA[1]:.3f}  '
          f'max={aA[2]:.3f}  sim={aA[3]:.3f}')
    print(f'  q      pointwise: mean={aQ[0]:.3f}  min={aQ[1]:.3f}  '
          f'max={aQ[2]:.3f}  sim={aQ[3]:.3f}')


if __name__ == '__main__':
    main()
