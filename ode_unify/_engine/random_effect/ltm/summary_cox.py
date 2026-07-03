"""Port of random effect/ltm/summary_cox.m.

Cox-frailty LTM summary.  Width-26 storage matches MATLAB
(``q_q + q_0`` larger under this setting).  SE filter is
``se_all(:,1) ∈ (0, 0.1)``.
"""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main
from .visual import visual


def summary_cox(N=2000, rep=1000, data_setting=1, knots_setting='K4',
                root=None):
    if root is None:
        root = os.path.dirname(__file__)
    res_sub = 'cox_rec_rd'; res_prefix = 'res_cox_N'
    res_dir = os.path.join(root, 'res', res_sub)

    est_params = np.zeros((rep, 26))
    est_se_all = np.zeros((rep, 26))

    def paths(seed, ks):
        base = os.path.join(
            res_dir,
            f'{res_prefix}{N}_seed{seed}_setting{data_setting}_knots{ks}',
        )
        return base + '.npz', base + '_fish.npz'

    for seed in range(1, rep + 1):
        print(seed)
        res4, fish4 = paths(seed, 'K4')
        res3, fish3 = paths(seed, 'K3')
        if not (os.path.isfile(res4) or os.path.isfile(res3)):
            try:
                run_main(N, seed, data_setting, knots_setting, True, root=root)
            except Exception:
                run_main(N, seed, data_setting, 'K3', True, root=root)
        if os.path.isfile(fish4):
            out = np.load(res4); fish = np.load(fish4)
        elif os.path.isfile(fish3):
            out = np.load(res3); fish = np.load(fish3)
        else:
            continue
        est_r = out['est_r'].ravel()
        se_all = fish['se_all'].ravel()
        n_store = min(26, est_r.size - 1)
        est_params[seed - 1, :n_store] = est_r[1:1 + n_store]
        est_se_all[seed - 1, :se_all.size] = se_all[:26]

    idx = (est_se_all[:, 0] > 0) & (est_se_all[:, 0] < 0.1)
    est_params = est_params[idx]
    est_se_all = est_se_all[idx]
    est_beta = est_params[:, :2]
    est_se_beta = est_se_all[:, :2]

    true_beta = np.array([1.0, 1.0])
    print('beta bias:'); print(est_beta.mean(0) - true_beta)
    print('beta SE:');   print(est_beta.std(0, ddof=1))
    print('ESE:');       print(est_se_beta.mean(0))
    up = est_beta + 1.96 * est_se_beta
    low = est_beta - 1.96 * est_se_beta
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))

    pxt = np.arange(0.0, 2.6 + 1e-9, 0.01)
    pxq = np.arange(0.0, 14.0 + 1e-9, 0.05)
    rep_valid = int(idx.sum())
    all_a = np.zeros((rep_valid, pxt.size))
    all_au = np.zeros((rep_valid, pxt.size))
    all_al = np.zeros((rep_valid, pxt.size))
    all_q = np.zeros((rep_valid, pxq.size))
    all_qu = np.zeros((rep_valid, pxq.size))
    all_ql = np.zeros((rep_valid, pxq.size))

    k = 0
    for seed in np.where(idx)[0] + 1:
        _, fish4 = paths(seed, 'K4')
        ks = 'K4' if os.path.isfile(fish4) else 'K3'
        print(seed)
        a, au, al, q, qu, ql = visual(
            N, seed, data_setting, ks, pxt, pxq, show=False, root=root,
        )
        a0 = a[0]
        all_a[k] = np.log(a / a0)
        all_au[k] = np.log(au / a0)
        all_al[k] = np.log(al / a0)
        all_q[k] = np.log(q * a0)
        all_qu[k] = np.log(qu * a0)
        all_ql[k] = np.log(ql * a0)
        k += 1

    try:
        import matplotlib.pyplot as plt
        true_a_log = np.log(pxt ** 2 + 1.0)
        true_q_log = np.zeros_like(pxq)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(pxt, true_a_log, 'b', lw=2, label='True Value')
        axes[0].plot(pxt, all_a.mean(0), 'r', lw=2,
                     label='Mean estimate(ODE-Flex)')
        axes[0].plot(pxt, all_au.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2,
                     label='95% Confidence Bound')
        axes[0].plot(pxt, all_al.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[0].set_xlim(0, 2); axes[0].set_ylim(-2, 3)
        axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$\log(\alpha(t))$')
        axes[0].legend(loc='upper left')
        axes[1].plot(pxq, true_q_log, 'b', lw=2)
        axes[1].plot(pxq, all_q.mean(0), 'r', lw=2)
        axes[1].plot(pxq, all_qu.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[1].plot(pxq, all_ql.mean(0), '-.',
                     color=(0.929, 0.694, 0.125), lw=2)
        axes[1].set_xlim(0, 10); axes[1].set_ylim(-2, 3)
        axes[1].set_xlabel('u'); axes[1].set_ylabel('q(u)')
        out_png = os.path.join(root, f'ltm_cox_rec_rd_{N}_rep{rep}.png')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        print(f'Wrote {out_png}')
    except Exception as exc:
        print(f'Plotting skipped: {exc}')


if __name__ == '__main__':
    summary_cox()
