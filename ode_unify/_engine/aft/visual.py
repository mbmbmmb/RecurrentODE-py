"""Port of aft/visual.m."""
from __future__ import annotations

import os
import numpy as np

from ..common import spcol


def visual(N, seed, data_setting, knots_setting, pxq, show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    se_file = os.path.join(
        root, 'res',
        f'res_aft_N{N}_seed{seed}_setting{data_setting}_knots{knots_setting}_se.npz',
    )
    res = np.load(se_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    p = int(res['p'].ravel()[0])
    knots = res['knots'].ravel()
    se_all = res['se_all'].ravel()

    est_theta = est_r[p:]
    se_theta = se_all[p:]

    pxq = np.asarray(pxq, dtype=float).ravel()
    B = spcol(knots, k, pxq)
    est_q = np.exp(B @ est_theta)
    est_q_upper = np.exp(B @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(B @ (est_theta - 1.96 * se_theta))

    if show:
        import matplotlib.pyplot as plt
        true_q = 2.0 / (1 + pxq)
        plt.figure()
        plt.plot(pxq, true_q, 'b', lw=2, label='True')
        plt.plot(pxq, est_q, 'r', lw=2, label='Estimate')
        plt.plot(pxq, est_q_upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        plt.plot(pxq, est_q_lower, '--', color=(0.929, 0.694, 0.125), lw=1,
                 label='95% CI')
        plt.title(f'Baseline Hazard Estimate for Seed {seed}')
        plt.xlabel('t'); plt.ylabel(r'$\lambda_0(t)$')
        plt.legend(loc='upper right')
        plt.grid(True)

    return est_q, est_q_upper, est_q_lower
