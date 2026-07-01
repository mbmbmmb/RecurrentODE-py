"""Port of random effect/cox/summary.m."""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary(N=2000, rep=1000, data_setting=1, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    est_r_all = np.zeros((rep, p))
    est_se = np.zeros((rep, p))
    est_time = np.zeros(rep)

    for seed in range(1, rep + 1):
        print(seed)
        res_seed = os.path.join(
            root, 'res',
            f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
        )
        if not os.path.isfile(res_seed):
            run_main(N, seed, data_setting, True, root=root)
        res_i = np.load(res_seed)
        est_r_all[seed - 1] = res_i['est_r'].ravel()[:p]
        est_se[seed - 1] = res_i['se_beta'].ravel()[:p]
        est_time[seed - 1] = float(res_i['runtime'].ravel()[0])

    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:'); print(est_r_all.mean(0) - true_beta)
    print('beta SE:'); print(est_r_all.std(0, ddof=1))
    print('ESE:'); print(est_se.mean(0))
    up = est_r_all + 1.96 * est_se
    low = est_r_all - 1.96 * est_se
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print(est_time.mean())

    px = np.arange(0.0, 1.8 + 1e-9, 0.01)
    all_est_a = np.zeros((rep, px.size))
    all_est_a_upper = np.zeros((rep, px.size))
    all_est_a_lower = np.zeros((rep, px.size))
    for seed in range(1, rep + 1):
        ea, eu, el = visual(N, seed, data_setting, px, show=False, root=root)
        all_est_a[seed - 1] = ea
        all_est_a_upper[seed - 1] = eu
        all_est_a_lower[seed - 1] = el

    try:
        import matplotlib.pyplot as plt
        true_a = px ** 2 + 1.0
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(px, true_a, 'b', lw=2, label='True')
        ax.plot(px, all_est_a.mean(0), 'r', lw=2, label='Estimate')
        ax.plot(px, all_est_a_upper.mean(0), '-.',
                color=(0.929, 0.694, 0.125), lw=2, label='95% CI')
        ax.plot(px, all_est_a_lower.mean(0), '-.',
                color=(0.929, 0.694, 0.125), lw=2)
        ax.set_xlabel('t'); ax.set_ylabel(r'$\lambda_0(t)$')
        ax.legend(loc='upper left')
        out_png = os.path.join(root, f'cox_rd_summary_{N}_rep{rep}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary()
