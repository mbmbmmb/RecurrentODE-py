"""Port of cox/inference.m."""
from __future__ import annotations

import os
import numpy as np

from .objective_func_inf import objective_func_inf


def inference(N, seed, data_setting, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_file = os.path.join(
        root, 'data', f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    result_file = os.path.join(
        root, 'res', f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    res = np.load(result_file)
    est_r = res['est_r'].ravel()
    k = int(res['k'].ravel()[0])
    knots = res['knots'].ravel()

    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    _, grad = objective_func_inf(est_r, x, time, delta, id_vec, knots, k)

    inv_fish = np.linalg.inv(grad.T @ grad)
    se_all = np.sqrt(np.diag(inv_fish))
    return se_all
