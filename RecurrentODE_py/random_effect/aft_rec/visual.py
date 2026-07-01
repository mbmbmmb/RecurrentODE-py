"""Port of random effect/aft_rec/visual.m."""
from __future__ import annotations

import os
import numpy as np

from ...common import spcol


def visual(N, seed, data_setting, knots_setting, px, show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    res_seed = os.path.join(
        root, 'res',
        f'res_aft_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}_se.npz',
    )
    res = np.load(res_seed)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    p = int(res['p'].ravel()[0])
    knots = res['knots'].ravel()
    se_all = res['se_all'].ravel()

    est_theta = est_r[p:]
    se_theta = se_all[p:]

    px = np.asarray(px, dtype=float).ravel()
    B = spcol(knots, k, px)
    est_q = np.exp(B @ est_theta)
    est_q_upper = np.exp(B @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(B @ (est_theta - 1.96 * se_theta))

    if show:
        import matplotlib.pyplot as plt
        true_q = 2.0 / (1.0 + px)
        fig, ax = plt.subplots()
        ax.plot(px, true_q, 'b', lw=2, label='True')
        ax.plot(px, est_q, 'r', lw=2, label='Estimate')
        ax.plot(px, est_q_upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.plot(px, est_q_lower, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.set_xlabel('u'); ax.set_ylabel('q(u)')
        ax.legend(loc='upper right')

    return est_q, est_q_upper, est_q_lower
