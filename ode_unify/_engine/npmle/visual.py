"""Port of npmle/visual.m.

The MATLAB version has a typo (``0.2./(1+px)`` — ``px`` is undefined); the
intent is ``0.2 / (1 + pxq)`` which matches the Box-Cox baseline used in
data_setting 3.
"""
from __future__ import annotations

import os
import numpy as np

from ..common import spcol


def visual(N, seed, data_setting, knots_setting, pxq, show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    res_file = os.path.join(
        root, 'res',
        f'res_Gtransform_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}_se.npz',
    )
    res = np.load(res_file)
    est_r = res['est_r'].ravel()
    se_all = res['se_all'].ravel()
    k = int(res['k'].ravel()[0])
    knots = res['knots'].ravel()
    p = int(res['p'].ravel()[0])

    est_theta = est_r[p:]
    se_theta = se_all[p:]

    pxq = np.asarray(pxq, dtype=float).ravel()
    B = spcol(knots, k, pxq)
    est_q = np.exp(B @ est_theta)
    est_q_upper = np.exp(B @ (est_theta + 1.96 * se_theta))
    est_q_lower = np.exp(B @ (est_theta - 1.96 * se_theta))

    if show:
        import matplotlib.pyplot as plt
        true_q = 0.2 / (1.0 + pxq)
        fig, ax = plt.subplots()
        ax.plot(pxq, true_q, 'b', lw=2, label='True')
        ax.plot(pxq, est_q, 'r', lw=2, label='Estimate')
        ax.plot(pxq, est_q_upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.plot(pxq, est_q_lower, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.set_xlabel('u'); ax.set_ylabel('q(u)')
        ax.legend(loc='upper right'); ax.grid(True)

    return est_q, est_q_upper, est_q_lower
