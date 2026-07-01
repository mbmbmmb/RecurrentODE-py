"""Port of random effect/cox/visual.m."""
from __future__ import annotations

import os
import numpy as np

from ...common import spcol
from .inference import inference


def visual(N, seed, data_setting, px, show=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    res_seed = os.path.join(
        root, 'res',
        f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
    )
    res = np.load(res_seed)
    est_r = res['est_r'].ravel()
    p = int(res['p'].ravel()[0])
    k = int(res['k'].ravel()[0])
    knots = res['knots'].ravel()

    inf_seed = os.path.join(
        root, 'res',
        f'res_cox_N{N}_seed{seed}_setting{data_setting}_inference.npz',
    )
    if not os.path.isfile(inf_seed):
        inference(N, seed, data_setting, root=root)
    se_all = np.load(inf_seed)['se_all'].ravel()

    est_theta = est_r[p:]
    se_theta = se_all[p:]

    px = np.asarray(px, dtype=float).ravel()
    B = spcol(knots, k, px)
    est_a = np.exp(B @ est_theta)
    est_a_upper = np.exp(B @ (est_theta + 1.96 * se_theta))
    est_a_lower = np.exp(B @ (est_theta - 1.96 * se_theta))

    if show:
        import matplotlib.pyplot as plt
        true_a = px ** 2 + 1.0
        fig, ax = plt.subplots()
        ax.plot(px, true_a, 'b', lw=2, label='True')
        ax.plot(px, est_a, 'r', lw=2, label='Estimate')
        ax.plot(px, est_a_upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.plot(px, est_a_lower, '--', color=(0.929, 0.694, 0.125), lw=1)
        ax.set_title(f'Baseline Hazard Estimate for Seed {seed}')
        ax.set_xlabel('t'); ax.set_ylabel(r'$\lambda_0(t)$')
        ax.legend(loc='upper left')

    return est_a, est_a_upper, est_a_lower
