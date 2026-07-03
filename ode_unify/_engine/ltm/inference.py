"""Port of ltm/inference.m."""
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
        f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    res = np.load(result_file)
    est_r = res['est_r'].ravel()
    knots_0 = res['knots_0'].ravel()
    knots_q = res['knots_q'].ravel()
    k0 = int(res['k0'].ravel()[0])
    kq = int(res['kq'].ravel()[0])

    p = x.shape[1]
    q_q = len(knots_q) - kq

    beta = est_r[1:p]
    theta = est_r[p:p + q_q]
    alpha = est_r[p + q_q:]

    _, grad_beta = objective_func_beta(
        beta, x, time, delta, id_vec, theta, alpha,
        knots_0, knots_q, k0, kq, ci=True,
    )
    _, grad_sieve = objective_func_sieve(
        np.concatenate([theta, alpha]),
        x, time, delta, id_vec, beta,
        knots_0, knots_q, k0, kq, ci=True,
    )
    grad = np.concatenate([grad_beta, grad_sieve], axis=1)

    try:
        M = grad.T @ grad
        eigvals, V = np.linalg.eigh(M)
        diag_d = np.maximum(eigvals, 1.0)
        inv_fish = (V * (1.0 / diag_d)) @ V.T
        se_all = np.sqrt(np.abs(np.diag(inv_fish)))
    except np.linalg.LinAlgError as exc:
        print('Could not compute standard errors.')
        print(exc)
        se_all = np.full(grad.shape[1], np.nan)
    return se_all
