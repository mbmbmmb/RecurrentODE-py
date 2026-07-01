"""Port of npmle/summary.m."""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary(N=1000, rep=1000, data_setting=3, knots_setting='equal', root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    max_cols = 11
    res = np.zeros((rep, max_cols))
    se_res = np.zeros((rep, max_cols))

    for seed in range(1, rep + 1):
        print(seed)
        run_main(N, seed, data_setting, knots_setting, True, root=root)
        out = np.load(os.path.join(
            root, 'res',
            f'res_Gtransform_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        ))
        est_r = out['est_r'].ravel()
        se_all = out['se_all'].ravel()
        res[seed - 1, :len(est_r)] = est_r
        se_res[seed - 1, :len(se_all)] = se_all

    est_beta = res[:, :p]
    se_beta = se_res[:, :p]
    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:'); print(est_beta.mean(0) - true_beta)
    print(est_beta.std(0, ddof=1))
    print('ESE:'); print(se_beta.mean(0))
    up = est_beta + 1.96 * se_beta
    low = est_beta - 1.96 * se_beta
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print('Gtransform n=1000 rep=1000 no rd effs')

    tol = 200
    pxq = np.linspace(0.01, 3.0, tol)
    est_curves = np.zeros((rep, tol))
    upper_ci = np.zeros((rep, tol))
    lower_ci = np.zeros((rep, tol))
    for seed in range(1, rep + 1):
        eq, eu, el = visual(
            N, seed, data_setting, knots_setting, pxq, show=False, root=root,
        )
        est_curves[seed - 1] = eq
        upper_ci[seed - 1] = eu
        lower_ci[seed - 1] = el

    try:
        import matplotlib.pyplot as plt
        true_curve = 0.2 / (pxq + 1.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pxq, true_curve, 'b', lw=3.5, label='True value')
        ax.plot(pxq, est_curves.mean(0), 'r', lw=2, label='Mean Estimate')
        ax.plot(pxq, upper_ci.mean(0), '--',
                color=(0.929, 0.694, 0.125), lw=2,
                label='95% Confidence Interval')
        ax.plot(pxq, lower_ci.mean(0), '--',
                color=(0.929, 0.694, 0.125), lw=2)
        ax.set_xlim(0, pxq.max())
        ax.set_xlabel('u'); ax.set_ylabel('q(u)')
        ax.legend(fontsize=14)
        out_png = os.path.join(root, f'npmle_ode_summary_{rep}_generator.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary()
