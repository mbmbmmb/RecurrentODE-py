"""Port of random effect/aft_rec/summary.m."""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary(N=2000, rep=1000, data_setting=2, knots_setting='equal', root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    est_r_all = np.zeros((rep, p))
    est_se = np.zeros((rep, p))
    runtime = np.zeros(rep)

    for seed in range(1, rep + 1):
        print(seed)
        res_seed = os.path.join(
            root, 'res',
            f'res_aft_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        if not os.path.isfile(res_seed):
            run_main(N, seed, data_setting, knots_setting, True, root=root)
        temp = np.load(res_seed)
        est_r_all[seed - 1] = temp['est_r'].ravel()[:p]
        est_se[seed - 1] = temp['se_all'].ravel()[:p]
        runtime[seed - 1] = float(temp['runtime'].ravel()[0])

    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:'); print(est_r_all.mean(0) - true_beta)
    print('beta SE:'); print(est_r_all.std(0, ddof=1))
    print('ESE:'); print(est_se.mean(0))
    up = est_r_all + 1.96 * est_se
    low = est_r_all - 1.96 * est_se
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print(np.vstack([est_r_all.mean(0) - true_beta, est_r_all.std(0, ddof=1),
                     est_se.mean(0), cp.mean(0)]).T)
    print(runtime.mean())

    pxq = np.arange(0.0, 4.0 + 1e-9, 0.05)
    lenq = pxq.size
    est_curves = np.zeros((rep, lenq))
    upper_ci = np.zeros((rep, lenq))
    lower_ci = np.zeros((rep, lenq))
    for seed in range(1, rep + 1):
        print(seed)
        te, tu, tl = visual(
            N, seed, data_setting, knots_setting, pxq, show=False, root=root,
        )
        est_curves[seed - 1] = te
        upper_ci[seed - 1] = tu
        lower_ci[seed - 1] = tl

    try:
        import matplotlib.pyplot as plt
        true_curve = 2.0 / (1.0 + pxq)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pxq, true_curve, 'b', lw=2, label='True value')
        ax.plot(pxq, est_curves.mean(0), 'r', lw=2, label='Mean Estimate')
        ax.plot(pxq, upper_ci.mean(0), '--',
                color=(0.929, 0.694, 0.125), lw=2,
                label='95% Confidence Interval')
        ax.plot(pxq, lower_ci.mean(0), '--',
                color=(0.929, 0.694, 0.125), lw=2)
        ax.set_xlim(0, pxq.max()); ax.set_ylim(0, 2.5)
        ax.set_xlabel('u'); ax.set_ylabel('q(u)'); ax.legend(fontsize=14)
        out_png = os.path.join(root, f'aft_ode_with_random_effects_summary_{N}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary()
