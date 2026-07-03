"""Port of ltm/summary.m."""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary(N=1000, data_setting=4, knots_setting='K4', rep=100, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    p = 3
    est_params = np.zeros((rep, 2))
    est_se = np.zeros((rep, 2))
    est_time = np.zeros(rep)
    err = np.zeros(rep)

    for seed in range(1, rep + 1):
        print(seed)
        res_seed = os.path.join(
            root, 'res',
            f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        used_knots = knots_setting
        try:
            run_main(N, seed, data_setting, knots_setting, True, root=root)
            temp = np.load(res_seed)
        except Exception:
            err[seed - 1] = 1
            used_knots = 'K3'
            run_main(N, seed, data_setting, used_knots, True, root=root)
            res_seed = os.path.join(
                root, 'res',
                f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
                f'_knots{used_knots}_se.npz',
            )
            temp = np.load(res_seed)
        est_r = temp['est_r'].ravel()
        se_all = temp['se_all'].ravel()
        runtime = float(temp['runtime'].ravel()[0])
        est_params[seed - 1] = est_r[1:3]
        est_se[seed - 1] = se_all[:2]
        est_time[seed - 1] = runtime

    true_beta = np.array([1.0, 1.0])
    print('beta bias:'); print(est_params.mean(0) - true_beta)
    print('beta SE:');   print(est_params.std(0, ddof=1))
    print('ESE:');       print(est_se.mean(0))
    up = est_params + 1.96 * est_se
    low = est_params - 1.96 * est_se
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))

    pxt = np.arange(0.0, 3.0 + 1e-9, 0.01)
    pxq = np.arange(0.0, 3.0 + 1e-9, 0.01)
    len2 = len(pxt); len1 = len(pxq)
    all_est_a = np.zeros((rep, len1))
    all_est_a_upper = np.zeros((rep, len1))
    all_est_a_lower = np.zeros((rep, len1))
    all_est_q = np.zeros((rep, len2))
    all_est_q_upper = np.zeros((rep, len2))
    all_est_q_lower = np.zeros((rep, len2))

    for seed in range(1, rep + 1):
        res_seed = os.path.join(
            root, 'res',
            f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        used_knots = knots_setting
        if not os.path.isfile(res_seed):
            used_knots = 'K3'
        e_a, e_au, e_al, e_q, e_qu, e_ql = visual(
            N, seed, data_setting, used_knots, pxq, pxt, show=False, root=root,
        )
        all_est_a[seed - 1] = e_a
        all_est_a_upper[seed - 1] = e_au
        all_est_a_lower[seed - 1] = e_al
        all_est_q[seed - 1] = e_q
        all_est_q_upper[seed - 1] = e_qu
        all_est_q_lower[seed - 1] = e_ql

    try:
        import matplotlib.pyplot as plt
        true_a = pxt + 1
        true_q = 2.0 / (1 + pxq)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(pxt, true_a, 'b', lw=2, label='True Value')
        axes[0].plot(
            pxt[:-1], all_est_a[:, :-1].mean(0), 'r', lw=2,
            label='Mean estimate(ODE-Flex)',
        )
        axes[0].plot(pxt[:-1], all_est_a_upper[:, :-1].mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2,
                     label='95% Confidence Bound')
        axes[0].plot(pxt[:-1], all_est_a_lower[:, :-1].mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[0].set_xlim(0, 2); axes[0].set_ylim(0, 6)
        axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$\alpha(t)$')
        axes[1].plot(pxq, true_q, 'b', lw=2)
        axes[1].plot(pxq, all_est_q.mean(0), 'r', lw=2)
        axes[1].plot(pxq, all_est_q_upper.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[1].plot(pxq, all_est_q_lower.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[1].set_ylim(0, 4.5)
        axes[1].set_xlabel('u'); axes[1].set_ylabel('q(u)')
        axes[0].legend(loc='upper left')
        out_png = os.path.join(root, f'ltm_alpha_{N}_rep_{rep}_setting{data_setting}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary()
