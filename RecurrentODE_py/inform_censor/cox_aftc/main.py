"""Driver for the AFT-censoring Cox recurrent-event simulation.

Identical fit/inference code as ``RecurrentODE_py.cox.main`` -- we only
swap the data generator for ``inform_censor.cox_aftc.generator_rec`` so
that censoring is drawn from an AFT-type survival model.
"""
from __future__ import annotations

import os
import time as _time
import numpy as np
from scipy.optimize import minimize

from ...common import augknt, ensure_dir
from ...cox.objective_func import objective_func
from ...cox.inference import inference as _cox_inference
from .generator_rec import generator_rec


def main(N, seed, data_setting, calculate_ci=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir = os.path.join(root, 'data')
    res_dir = os.path.join(root, 'res')
    ensure_dir(data_dir); ensure_dir(res_dir)

    data_file = os.path.join(
        data_dir, f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    if not os.path.isfile(data_file):
        generator_rec(N, seed, data_setting, data_dir=data_dir)
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()

    p = x.shape[1]
    k = 3
    l = x.shape[0] ** (1.0 / 7.0)
    num_breaks = int(np.floor(l)) + 1
    knots = augknt(np.linspace(0.0, float(np.max(time)), num_breaks), k)
    q = len(knots) - k

    def obj_and_grad(params):
        return objective_func(params, x, time, delta, knots, k)

    x0 = np.zeros(p + q)
    t0 = _time.time()
    result = minimize(
        obj_and_grad, x0,
        jac=True, method='BFGS',
        options={'maxiter': 500, 'gtol': 1e-6},
    )
    runtime = _time.time() - t0
    est_r = result.x

    out_path = os.path.join(
        res_dir, f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        est_r=est_r.reshape(-1, 1),
        runtime=runtime,
        k=k, l=l, p=p, q=q, knots=knots,
    )

    if calculate_ci:
        se_all = _cox_inference(N, seed, data_setting, root=root)
        out_path_se = os.path.join(
            res_dir,
            f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
        )
        np.savez_compressed(
            out_path_se,
            est_r=est_r.reshape(-1, 1),
            se_all=se_all.reshape(-1, 1),
            k=k, l=l, p=p, q=q, knots=knots,
        )
    return est_r, runtime


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if len(args) > 0 else 1000
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 1
    calculate_ci = bool(int(args[3])) if len(args) > 3 else False
    est, rt = main(N, seed, data_setting, calculate_ci)
    print('est:', est)
    print('runtime:', rt)
