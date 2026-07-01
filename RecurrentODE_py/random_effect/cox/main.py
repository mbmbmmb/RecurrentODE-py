"""Port of random effect/cox/main.m."""
from __future__ import annotations

import os
import time as _time
import numpy as np
from scipy.optimize import minimize

from ...common import augknt, ensure_dir
from .generator_rec import generator_rec
from .objective_func import objective_func
from .inference_beta import inference_beta


def main(N, seed, data_setting, ci=False, root=None):
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
    knots = augknt(np.linspace(0.0, float(np.max(time)), int(l) + 1), k)
    q = knots.size - k

    def fun(r):
        return objective_func(r, x, time, delta, knots, k)

    r0 = np.zeros(p + q)
    t0 = _time.time()
    res_opt = minimize(
        fun, r0, jac=True, method='BFGS',
        options={'maxiter': 500},
    )
    runtime = _time.time() - t0
    est_r = res_opt.x
    print(est_r[:p])

    out_path = os.path.join(
        res_dir, f'res_cox_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        est_r=est_r.reshape(-1, 1), runtime=runtime,
        p=p, q=q, k=k, l=l, knots=knots,
    )

    if ci:
        fish = inference_beta(N, seed, data_setting, root=root)
        se_beta = np.sqrt(np.abs(np.diag(fish)))
        out_path_se = os.path.join(
            res_dir,
            f'res_cox_N{N}_seed{seed}_setting{data_setting}_se.npz',
        )
        np.savez_compressed(
            out_path_se,
            est_r=est_r.reshape(-1, 1), runtime=runtime,
            p=p, q=q, k=k, l=l, knots=knots,
            se_beta=se_beta.reshape(-1, 1),
        )
    return est_r, runtime


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 1
    ci = bool(int(args[3])) if len(args) > 3 else False
    main(N, seed, data_setting, ci)
