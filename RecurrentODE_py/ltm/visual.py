"""Port of ltm/visual.m."""
from __future__ import annotations

import os
import numpy as np

from ..common import spcol


def visual(N, seed, data_setting, knots_setting, pxt, pxq, show=False,
           root=None):
    if root is None:
        root = os.path.dirname(__file__)
    print(seed)
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    data = np.load(data_file)
    x = data['x']
    p = x.shape[1]

    result_file = os.path.join(
        root, 'res',
        f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}_se.npz',
    )
    res = np.load(result_file)
    est_r = res['est_r'].ravel()
    se_all = res['se_all'].ravel()
    knots_0 = res['knots_0'].ravel()
    knots_q = res['knots_q'].ravel()
    k0 = int(res['k0'].ravel()[0])
    kq = int(res['kq'].ravel()[0])
    q_0 = int(res['q_0'].ravel()[0])
    q_q = len(knots_q) - kq

    pxt = np.asarray(pxt, dtype=float).ravel()
    pxq = np.asarray(pxq, dtype=float).ravel()

    est_theta = est_r[p:p + q_q]
    se_theta = se_all[p - 1:p - 1 + q_q]
    est_alpha = est_r[p + q_q:p + q_q + q_0]
    se_alpha = se_all[p - 1 + q_q:p - 1 + q_q + q_0]

    Bq = spcol(knots_q, kq, pxq)
    est_q = np.exp(Bq @ est_theta)
    est_q_upper = np.exp(Bq @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(Bq @ (est_theta - 1.96 * se_theta))

    B0 = spcol(knots_0, k0, pxt)
    est_a = np.exp(B0 @ est_alpha)
    est_a_upper = np.exp(B0 @ (est_alpha + 1.96 * se_alpha))
    est_a_lower = np.exp(B0 @ (est_alpha - 1.96 * se_alpha))

    if data_setting == 1:
        true_a = pxt ** 3; true_q = np.ones_like(pxq)
        est_a = est_a * 1.5 ** 3; est_q = est_q / 1.5 ** 3
    elif data_setting == 2:
        true_a = np.full_like(pxt, 2.0); true_q = np.exp(-pxq)
        est_a = est_a * 2; est_q = est_q / 2
    elif data_setting == 3:
        true_a = np.ones_like(pxt); true_q = 2.0 / (1 + pxq)
    elif data_setting == 4:
        true_a = pxt + 1; true_q = 2.0 / (1 + pxq)
        est_a = 3 * est_a
        est_a_upper = 3 * est_a_upper
        est_a_lower = 3 * est_a_lower
        est_q = est_q / 3
        est_q_upper = est_q_upper / 3
        est_q_lower = est_q_lower / 3

    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(pxt, true_a, 'b', lw=2, label='True')
        axes[0].plot(pxt, est_a, 'r', lw=2, label='Estimate')
        axes[0].plot(pxt, est_a_upper, '-.', color=(0.929, 0.694, 0.125), lw=2)
        axes[0].plot(pxt, est_a_lower, '-.', color=(0.929, 0.694, 0.125), lw=2)
        axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$\alpha(t)$')

        axes[1].plot(pxq, true_q, 'b', lw=2)
        axes[1].plot(pxq, est_q, 'r', lw=2)
        axes[1].plot(pxq, est_q_upper, '-.', color=(0.929, 0.694, 0.125), lw=2)
        axes[1].plot(pxq, est_q_lower, '-.', color=(0.929, 0.694, 0.125), lw=2)
        axes[1].set_xlabel('u'); axes[1].set_ylabel('q(u)')

    return est_a, est_a_upper, est_a_lower, est_q, est_q_upper, est_q_lower
