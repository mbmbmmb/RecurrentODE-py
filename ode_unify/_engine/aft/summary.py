"""Port of aft/summary.m.

Summary script covering (a) coverage of the 95% CIs for beta, (b) functional
parameter empirical CIs across replications, and (c) runtime vs. sample size.
The R-based ``reReg`` comparison numbers are taken verbatim from the MATLAB
script.
"""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def coverage_summary(N=1000, data_setting=2, knots_setting='equal', rep=1000,
                     root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    est_beta = np.zeros((rep, p))
    est_se = np.zeros((rep, p))
    est_time = np.zeros(rep)
    for seed in range(1, rep + 1):
        se_file = os.path.join(
            root, 'res',
            f'res_aft_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        res_file = os.path.join(
            root, 'res',
            f'res_aft_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}.npz',
        )
        if not os.path.isfile(se_file):
            run_main(N, seed, data_setting, knots_setting, True, root=root)
        out = np.load(se_file)
        est_beta[seed - 1] = out['est_r'].ravel()[:p]
        est_se[seed - 1] = out['se_all'].ravel()[:p]
        est_time[seed - 1] = float(np.load(res_file)['runtime'].ravel()[0])

    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:'); print(est_beta.mean(0) - true_beta)
    print('beta SE:');   print(est_beta.std(0, ddof=1))
    print('ESE:');       print(est_se.mean(0))
    up = est_beta + 1.96 * est_se
    low = est_beta - 1.96 * est_se
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print(est_time.mean())


def functional_summary(N=1000, data_setting=2, knots_setting='equal', rep=1000,
                       root=None):
    if root is None:
        root = os.path.dirname(__file__)
    pxq = np.arange(0.0, 4.0 + 1e-9, 0.05)
    lenq = len(pxq)
    est_curves = np.zeros((rep, lenq))
    upper = np.zeros((rep, lenq))
    lower = np.zeros((rep, lenq))
    for seed in range(1, rep + 1):
        se_file = os.path.join(
            root, 'res',
            f'res_aft_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        if not os.path.isfile(se_file):
            run_main(N, seed, data_setting, knots_setting, True, root=root)
        e, u, l = visual(N, seed, data_setting, knots_setting, pxq,
                         show=False, root=root)
        est_curves[seed - 1] = e; upper[seed - 1] = u; lower[seed - 1] = l

    try:
        import matplotlib.pyplot as plt
        true_curve = 2.0 / (1 + pxq)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pxq, true_curve, 'b', lw=2, label='True value')
        ax.plot(pxq, est_curves.mean(0), 'r', lw=2, label='Mean Estimate')
        ax.plot(pxq, upper.mean(0), '--', color=(0.929, 0.694, 0.125), lw=2,
                label='95% Confidence Interval')
        ax.plot(pxq, lower.mean(0), '--', color=(0.929, 0.694, 0.125), lw=2)
        ax.set_xlim(0, pxq.max()); ax.set_ylim(0, 2.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel('u'); ax.set_ylabel('q(u)')
        ax.legend(fontsize=16)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(16)
        out_dir = os.path.join(root, 'plot'); os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f'aft_ode_summary_{rep}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


def time_comparison(data_setting=2, knots_setting='equal', rep=100, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    Ns = [1000, 2000, 4000, 8000]
    est_time = np.zeros((rep, 4))
    for i, N in enumerate(Ns):
        for seed in range(1, rep + 1):
            res_seed = os.path.join(
                root, 'res',
                f'res_aft_N{N}_seed{seed}_setting{data_setting}'
                f'_knots{knots_setting}.npz',
            )
            run_main(N, seed, data_setting, knots_setting, False, root=root)
            est_time[seed - 1, i] = float(
                np.load(res_seed)['runtime'].ravel()[0]
            )
    run_time = est_time / est_time[:, 0].mean()
    time_err = run_time.std(0, ddof=1)
    time_mean = run_time.mean(0)

    res_rereg = np.array([1.000000, 2.644014, 8.004913, 48.671311])
    res_rereg_std = np.array([0.2500155, 0.6179043, 1.8129716, 5.6981323])

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.errorbar(Ns, time_mean, time_err, lw=1, label='ODE-AM')
        ax.errorbar(Ns, res_rereg, res_rereg_std, color='r', lw=1,
                    label='reReg-am.GL')
        xref = np.arange(1000, 8001)
        ax.plot(xref, xref / 1000, '--', color=(0.466, 0.674, 0.188), lw=1,
                label='Reference O(N)')
        ax.plot(xref, (xref ** 2) / (1000 ** 2), '--',
                color=(0.929, 0.694, 0.125), lw=1, label='Reference O(N^2)')
        ax.set_xlim(900, 8500); ax.set_ylim(0.5, 100)
        ax.set_xticks(Ns); ax.set_xticklabels(['1', '2', '4', '8'])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'Sample size ($\times 10^3$)')
        ax.set_ylabel('Relative computing time')
        ax.legend(loc='upper left')
        out_png = os.path.join(root, 'ode_am_baseline_time_comparison.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    coverage_summary()
    functional_summary()
