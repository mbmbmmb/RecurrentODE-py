"""Port of npmle/inference.m."""
from __future__ import annotations

import os
import numpy as np

from .objective_func import objective_func


def inference(N, seed, data_setting, knots_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    res_file = os.path.join(
        root, 'res',
        f'res_Gtransform_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()
    rho1 = float(np.asarray(data['rho1']).ravel()[0])
    r1 = float(np.asarray(data['r1']).ravel()[0])

    res = np.load(res_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    q = int(res['q'].ravel()[0])
    knots = res['knots'].ravel()

    _, grad = objective_func(
        est_r, x, time, delta, id_vec, q, knots, k, rho1, r1, ci=True,
    )
    num_params = grad.shape[1]
    is_nonzero_col = np.any(grad != 0.0, axis=0)
    grad_reduced = grad[:, is_nonzero_col]

    try:
        inv_fish_reduced = np.linalg.inv(grad_reduced.T @ grad_reduced)
        se_reduced = np.sqrt(np.abs(np.diag(inv_fish_reduced)))
        se_all = np.zeros(num_params)
        se_all[is_nonzero_col] = se_reduced
    except np.linalg.LinAlgError as exc:
        print('Matrix is singular or close to singular. Could not compute standard errors.')
        print(exc)
        se_all = np.full(num_params, np.nan)
    return se_all
