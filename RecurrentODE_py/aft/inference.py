"""Port of aft/inference.m."""
from __future__ import annotations

import os
import numpy as np

from .objective_func_beta import objective_func_beta
from .objective_func_sieve import objective_func_sieve


def inference(N, seed, data_setting, knots_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    result_file = os.path.join(
        root, 'res',
        f'res_aft_N{N}_seed{seed}_setting{data_setting}_knots{knots_setting}.npz',
    )
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    res = np.load(result_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    p = int(res['p'].ravel()[0])
    knots = res['knots'].ravel()

    forward = True
    beta = est_r[:p]
    theta = est_r[p:]

    _, grad_beta = objective_func_beta(
        beta, x, time, delta, id_vec, theta, knots, k, ci=True,
    )
    _, grad_sieve = objective_func_sieve(
        theta, x, time, delta, id_vec, beta, knots, k, ci=True, forward=forward,
    )
    grad = np.concatenate([grad_beta, grad_sieve], axis=1)

    is_nonzero_col = np.any(grad != 0, axis=0)
    grad_reduced = grad[:, is_nonzero_col]

    try:
        inv_fish_reduced = np.linalg.inv(grad_reduced.T @ grad_reduced)
        se_reduced = np.sqrt(np.diag(inv_fish_reduced))
        se_all = np.zeros(grad.shape[1])
        se_all[is_nonzero_col] = se_reduced
    except np.linalg.LinAlgError as exc:
        print('Matrix is singular; returning NaN standard errors.')
        print(exc)
        se_all = np.full(grad.shape[1], np.nan)
    return se_all
