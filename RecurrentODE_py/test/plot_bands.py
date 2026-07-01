"""Save per-module functional-parameter coverage demo plots.

For each module/setting, reads the persisted per-seed ``.npz`` files in
``test/results/<slug>/``, constructs the pointwise Wald band
``exp(B(u) @ (theta_hat +/- 1.96 * se_theta))`` on a grid, averages
across replications, and saves a plot of

- the true curve,
- the mean estimate,
- the mean 95 % band,

to ``test/band_plots/<slug>.png``.

For LTM / RE-LTM, the identifiability scale is pinned before
comparing to the truth (plain LTM uses a setting-dependent constant
scale factor, matching ``ltm/visual.py``; RE-LTM uses per-replication
alpha(0) normalisation, matching ``random_effect/ltm/visual.py``).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RecurrentODE_py.common import spcol


RESULTS = os.path.join(os.path.dirname(__file__), 'results')
PLOTS   = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS, exist_ok=True)


def _iter_seed_files(slug):
    d = os.path.join(RESULTS, slug)
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if n.startswith('seed') and n.endswith('.npz')]


def _coverage(est, up, lo, true_curve):
    inside = (lo <= true_curve[None, :]) & (true_curve[None, :] <= up)
    pw = inside.mean(axis=0)
    sim = inside.all(axis=1).mean()
    return pw.mean(), pw.min(), pw.max(), float(sim)


def _plot(title, xlabel, ylabel, grid, true_curve, est, up, lo, outfile,
          cov_str=None, ylim=None, use_median=False):
    """Display `avg` line across seeds as mean (default) or median.

    For RE-LTM a few seeds can produce very large spline SEs, which makes
    the arithmetic mean of ``exp(B * (theta + 1.96*se))`` unusable on a
    linear axis; median is robust. Coverage stats are independent of this
    choice (they are computed per-seed, then averaged 0/1).
    """
    agg = np.median if use_median else np.mean
    label_prefix = 'Median' if use_median else 'Mean'
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(grid, true_curve, 'b-', lw=2, label='True curve')
    ax.plot(grid, agg(est, axis=0), 'r-', lw=1.5, label=f'{label_prefix} estimate')
    ax.plot(grid, agg(up, axis=0), '--', color=(0.929, 0.694, 0.125), lw=1.2,
            label=f'{label_prefix} 95 % upper')
    ax.plot(grid, agg(lo, axis=0), '--', color=(0.929, 0.694, 0.125), lw=1.2,
            label=f'{label_prefix} 95 % lower')
    if ylim: ax.set_ylim(ylim)
    ax.set_title(title + (f'\n{cov_str}' if cov_str else ''))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f'  wrote {outfile}')


# ---------- single-spline modules ----------

def single_spline(slug, truth_fn, grid, title, xlabel, ylabel, outfile):
    est_all, up_all, lo_all = [], [], []
    for path in _iter_seed_files(slug):
        f = np.load(path)
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        p = int(f['p'].ravel()[0]); k = int(f['k'].ravel()[0])
        knots = f['knots'].ravel()
        theta = est_r[p:]; se = se_all[p:]
        B = spcol(knots, k, grid)
        est_all.append(np.exp(B @ theta))
        up_all.append(np.exp(B @ (theta + 1.96 * se)))
        lo_all.append(np.exp(B @ (theta - 1.96 * se)))
    est = np.asarray(est_all); up = np.asarray(up_all); lo = np.asarray(lo_all)
    tc = truth_fn(grid)
    m, mn, mx, sim = _coverage(est, up, lo, tc)
    cov = f'pointwise cov={m:.3f} (min {mn:.2f}, max {mx:.2f})  sim={sim:.2f}  n={len(est)}'
    _plot(title, xlabel, ylabel, grid, tc, est, up, lo,
          os.path.join(PLOTS, outfile), cov_str=cov)


# ---------- LTM (plain) — constant rescale ----------

def ltm_plot(slug, data_setting, grid_t, grid_u, title_pair,
             scale_a, scale_q, outfiles):
    A=[];Au=[];Al=[];Q=[];Qu=[];Ql=[]
    for path in _iter_seed_files(slug):
        f = np.load(path)
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        q_q = int(f['q_q']); q_0 = int(f['q_0'])
        p = (int(f['p']) if 'p' in f.files else est_r.size - q_q - q_0)
        kn0 = f['knots_0']; knq = f['knots_q']
        k0 = int(f['k0']); kq = int(f['kq'])
        theta = est_r[p:p+q_q]; se_th = se_all[p-1:p-1+q_q]
        alpha = est_r[p+q_q:p+q_q+q_0]; se_al = se_all[p-1+q_q:p-1+q_q+q_0]
        B0 = spcol(kn0, k0, grid_t); Bq = spcol(knq, kq, grid_u)
        A.append(scale_a * np.exp(B0 @ alpha))
        Au.append(scale_a * np.exp(B0 @ (alpha + 1.96*se_al)))
        Al.append(scale_a * np.exp(B0 @ (alpha - 1.96*se_al)))
        Q.append(scale_q * np.exp(Bq @ theta))
        Qu.append(scale_q * np.exp(Bq @ (theta + 1.96*se_th)))
        Ql.append(scale_q * np.exp(Bq @ (theta - 1.96*se_th)))
    A, Au, Al = map(np.asarray, (A, Au, Al))
    Q, Qu, Ql = map(np.asarray, (Q, Qu, Ql))
    # truth (setting-4 specific for plain LTM)
    true_a = grid_t + 1.0
    true_q = 2.0 / (1.0 + grid_u)
    ma = _coverage(A, Au, Al, true_a)
    mq = _coverage(Q, Qu, Ql, true_q)
    _plot(title_pair[0], 't', r'$\alpha(t)$', grid_t, true_a, A, Au, Al,
          os.path.join(PLOTS, outfiles[0]),
          cov_str=f'pw cov={ma[0]:.3f} (min {ma[1]:.2f})  sim={ma[3]:.2f}  n={len(A)}  (scale α={scale_a})')
    _plot(title_pair[1], 'u', 'q(u)', grid_u, true_q, Q, Qu, Ql,
          os.path.join(PLOTS, outfiles[1]),
          cov_str=f'pw cov={mq[0]:.3f} (min {mq[1]:.2f})  sim={mq[3]:.2f}  n={len(Q)}  (scale q={scale_q:.3f})')


# ---------- RE-LTM — MATLAB identifiability pin alpha(1.5) = 1 ----------

def re_ltm_plot(slug, data_setting, grid_t, grid_u, title_pair, outfiles,
                t_pin=1.5):
    """Matches MATLAB's fmincon constraint: alpha(t_pin) = 1.

    Since the estimator is constrained so alpha_hat(t_pin) = 1 exactly, the
    correct post-hoc rescale to compare with truth is simply the constant
    ``scale_a = alpha_true(t_pin)`` (and the reciprocal for q).
    """
    if data_setting == 1:      # Cox-frailty LTM, alpha(t)=t^2+1
        scale_a = t_pin ** 2 + 1.0
    else:                       # AFT-frailty LTM, alpha(t)=1
        scale_a = 1.0
    scale_q = 1.0 / scale_a
    A=[];Au=[];Al=[];Q=[];Qu=[];Ql=[]
    for path in _iter_seed_files(slug):
        f = np.load(path)
        est_r = f['est_r'].ravel(); se_all = f['se_all'].ravel()
        p = int(f['p']); q_q = int(f['q_q']); q_0 = int(f['q_0'])
        kn0 = f['knots_0']; knq = f['knots_q']
        k0 = int(f['k0']); kq = int(f['kq'])
        theta = est_r[p:p+q_q]; se_th = se_all[p-1:p-1+q_q]
        alpha = est_r[p+q_q:p+q_q+q_0]; se_al = se_all[p-1+q_q:p-1+q_q+q_0]
        B0 = spcol(kn0, k0, grid_t); Bq = spcol(knq, kq, grid_u)
        A.append(scale_a * np.exp(B0 @ alpha))
        Au.append(scale_a * np.exp(B0 @ (alpha + 1.96*se_al)))
        Al.append(scale_a * np.exp(B0 @ (alpha - 1.96*se_al)))
        Q.append(scale_q * np.exp(Bq @ theta))
        Qu.append(scale_q * np.exp(Bq @ (theta + 1.96*se_th)))
        Ql.append(scale_q * np.exp(Bq @ (theta - 1.96*se_th)))
    A, Au, Al = map(np.asarray, (A, Au, Al))
    Q, Qu, Ql = map(np.asarray, (Q, Qu, Ql))
    if data_setting == 1:      # Cox-frailty LTM
        true_a = grid_t ** 2 + 1.0; true_q = np.ones_like(grid_u)
    else:                       # AFT-frailty LTM
        true_a = np.ones_like(grid_t); true_q = 2.0 / (1.0 + grid_u)
    ma = _coverage(A, Au, Al, true_a)
    mq = _coverage(Q, Qu, Ql, true_q)
    _plot(title_pair[0], 't', r'$\alpha(t)$', grid_t, true_a, A, Au, Al,
          os.path.join(PLOTS, outfiles[0]),
          cov_str=f'pw cov={ma[0]:.3f} (min {ma[1]:.2f})  sim={ma[3]:.2f}  n={len(A)}  (pin α({t_pin})=1, ×{scale_a:.3f})',
          use_median=True)
    _plot(title_pair[1], 'u', 'q(u)', grid_u, true_q, Q, Qu, Ql,
          os.path.join(PLOTS, outfiles[1]),
          cov_str=f'pw cov={mq[0]:.3f} (min {mq[1]:.2f})  sim={mq[3]:.2f}  n={len(Q)}  (×{scale_q:.3f})',
          use_median=True)


# ---------- RE-cox band summary (already persisted) ----------

def re_cox_plot():
    p = os.path.join(RESULTS, 'random_effect_cox_setting1_band', '_summary.npz')
    if not os.path.isfile(p):
        print(f'  skip re_cox: {p} missing')
        return
    f = np.load(p)
    grid = f['grid']; est = f['est']; up = f['up']; lo = f['lo']
    true_curve = f['true_curve']
    m = _coverage(est, up, lo, true_curve)
    _plot('random_effect.cox / setting 1  (λ₀(t)=t²+1)',
          't', r'$\lambda_0(t)$', grid, true_curve, est, up, lo,
          os.path.join(PLOTS, 're_cox_s1.png'),
          cov_str=f'pw cov={m[0]:.3f} (min {m[1]:.2f})  sim={m[3]:.2f}  n={est.shape[0]}')


def main():
    # single-spline modules
    single_spline('cox_setting1',
                  lambda u: u ** 2 + 1, np.linspace(0.1, 2.5, 60),
                  'cox / setting 1  (q(u)=u²+1)',
                  't', r'$\lambda_0(t)$', 'cox_s1.png')
    single_spline('aft_setting2_knotsquantile',
                  lambda u: 2.0 / (1 + u), np.linspace(0.1, 3.5, 60),
                  'aft / setting 2  (q(u)=2/(1+u))',
                  'u', 'q(u)', 'aft_s2.png')
    single_spline('npmle_setting3_knotsequal',
                  lambda u: 0.2 / (1 + u), np.linspace(0.05, 2.5, 60),
                  'npmle / setting 3  (q(u)=0.2/(1+u))',
                  'u', 'q(u)', 'npmle_s3.png')
    single_spline('random_effect_aft_rec_setting2_knotsquantile',
                  lambda u: 2.0 / (1 + u), np.linspace(0.1, 3.5, 60),
                  'random_effect.aft_rec / setting 2  (q(u)=2/(1+u))',
                  'u', 'q(u)', 're_aft_s2.png')

    # plain LTM (setting 4)
    ltm_plot('ltm_setting4_knotsK4', 4,
             grid_t=np.linspace(0.05, 2.0, 60),
             grid_u=np.linspace(0.05, 2.0, 60),
             title_pair=('ltm / setting 4  (α(t)=t+1, scaled ×3)',
                         'ltm / setting 4  (q(u)=2/(1+u), scaled ×1/3)'),
             scale_a=3.0, scale_q=1.0 / 3.0,
             outfiles=('ltm_s4_alpha.png', 'ltm_s4_q.png'))

    # RE-LTM (setting 1) — MATLAB identifiability pin α(1.5) = 1
    re_ltm_plot('random_effect_ltm_setting1_knotsK4', 1,
                grid_t=np.linspace(0.01, 2.0, 60),
                grid_u=np.linspace(0.01, 10.0, 60),
                title_pair=(
                    'random_effect.ltm / setting 1  (α(t)=t²+1)',
                    'random_effect.ltm / setting 1  (q(u)=1)',
                ),
                outfiles=('re_ltm_s1_alpha.png', 're_ltm_s1_q.png'))

    # RE-cox
    re_cox_plot()


if __name__ == '__main__':
    main()
