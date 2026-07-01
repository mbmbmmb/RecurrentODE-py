"""Port of cox/summary.m.

Runs ``rep`` replications of ``main`` (generating any missing SE files) and
reports bias, empirical SE, model-based SE, coverage, and average runtime.
Produces the baseline-hazard coverage plot if matplotlib is available.
"""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary(N=1000, data_setting=1, rep=1000, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    est_params = np.zeros((rep, p))
    est_se = np.zeros((rep, p))
    est_time = np.zeros(rep)

    for seed in range(1, rep + 1):
        print(seed)
        se_file = os.path.join(
            root, 'res',
            f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
        )
        res_file = os.path.join(
            root, 'res',
            f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
        )
        if not os.path.isfile(se_file):
            run_main(N, seed, 1, True, root=root)
        out = np.load(res_file)
        se_mat = np.load(se_file)
        est_params[seed - 1] = out['est_r'].ravel()[:p]
        est_se[seed - 1] = se_mat['se_all'].ravel()[:p]
        est_time[seed - 1] = float(out['runtime'].ravel()[0])

    est_beta = est_params[:, :p]
    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:');  print(est_beta.mean(0) - true_beta)
    print('beta SE:');    print(est_beta.std(0, ddof=1))
    print('ESE:');        print(est_se.mean(0))

    up = est_beta + 1.96 * est_se
    low = est_beta - 1.96 * est_se
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print(est_time.mean())

    # --- functional parameter coverage plot ---
    px = np.arange(0.0, 3.0 + 1e-9, 0.01)
    len1 = len(px)
    est_curves = np.zeros((rep, len1))
    upper_ci = np.zeros((rep, len1))
    lower_ci = np.zeros((rep, len1))
    print(f'Running {rep} simulation replications...')
    for seed in range(1, rep + 1):
        e, u, l = visual(N, seed, data_setting, px, show=False, root=root)
        est_curves[seed - 1] = e
        upper_ci[seed - 1] = u
        lower_ci[seed - 1] = l
    print('Simulation complete.')

    try:
        import matplotlib.pyplot as plt
        true_curve = px ** 2 + 1
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(px, true_curve, 'b', lw=3.5, label='True value')
        ax.plot(px, est_curves.mean(0), 'r', lw=2, label='Mean Estimate')
        ax.plot(px, upper_ci.mean(0), '--', color=(0.929, 0.694, 0.125), lw=2,
                label='95% Confidence Interval')
        ax.plot(px, lower_ci.mean(0), '--', color=(0.929, 0.694, 0.125), lw=2)
        ax.set_xlim(0, px.max())
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel('u'); ax.set_ylabel('q(u)')
        ax.legend(fontsize=16, loc='upper left')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(16)
        out_png = os.path.join(root, f'cox_ode_summary_{rep}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary()
