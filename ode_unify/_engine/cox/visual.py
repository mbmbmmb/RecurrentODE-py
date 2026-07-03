"""Port of cox/visual.m.

Returns the estimated baseline hazard at grid ``px`` plus pointwise 95% CIs
derived from the stored SE file. If ``show`` is True, renders a matplotlib
figure; otherwise just returns the arrays.
"""
from __future__ import annotations

import os
import numpy as np

from ..common import spcol


def visual(N, seed, data_setting, px, show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    se_file = os.path.join(
        root, 'res', f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
    )
    se_mat = np.load(se_file)
    est_r = se_mat['est_r'].ravel()
    k = int(se_mat['k'].ravel()[0])
    knots = se_mat['knots'].ravel()
    se_all = se_mat['se_all'].ravel()

    x = np.load(data_file)['x']
    p = x.shape[1]

    est_theta = est_r[p:]
    se_theta = se_all[p:]

    px = np.asarray(px, dtype=float).ravel()
    B = spcol(knots, k, px)

    est_q = np.exp(B @ est_theta)
    est_q_upper = np.exp(B @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(B @ (est_theta - 1.96 * se_theta))

    if show:
        import matplotlib.pyplot as plt
        true_q = px ** 2 + 1
        plt.figure()
        plt.plot(px, true_q, 'b', lw=2, label='True')
        plt.plot(px, est_q, 'r', lw=2, label='Estimate')
        plt.plot(px, est_q_upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        plt.plot(px, est_q_lower, '--', color=(0.929, 0.694, 0.125), lw=1,
                 label='95% CI')
        plt.title(f'Baseline Hazard Estimate for Seed {seed}')
        plt.xlabel('t'); plt.ylabel(r'$\lambda_0(t)$')
        plt.legend(loc='upper left')
        plt.grid(True)

    return est_q, est_q_upper, est_q_lower
