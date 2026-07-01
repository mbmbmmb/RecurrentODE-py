"""Port of npmle/main.m."""
from __future__ import annotations

import os
import time as _time
import numpy as np
from scipy.optimize import minimize

from ..common import augknt, ensure_dir
from .generator_rec import generator_rec
from .objective_func import objective_func
from .inference import inference


def main(N, seed, data_setting, knots_setting='equal', ci=False, root=None):
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
    id_vec = data['id'].ravel()
    rho1 = float(np.asarray(data['rho1']).ravel()[0])
    r1 = float(np.asarray(data['r1']).ravel()[0])

    N_subj, p = x.shape
    k = 3  # quadratic
    l = int(np.ceil(N ** (1.0 / 5.0)) + 2)
    temp = time[delta > 0]
    if knots_setting == 'quantile':
        interior = np.quantile(temp, np.arange(1, l) / l)
        knots = augknt(
            np.concatenate([[0.0], interior, [float(np.max(time))]]), k,
        )
    elif knots_setting == 'equal':
        knots = augknt(np.linspace(0.0, float(np.max(time)), l + 1), k)
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')
    q = knots.size - k

    def fun(r):
        return objective_func(
            r, x, time, delta, id_vec, q, knots, k, rho1, r1, ci=False,
        )

    r0 = np.zeros(p + q)
    t0 = _time.time()
    res_opt = minimize(
        fun, r0, jac=True, method='BFGS',
        options={'maxiter': 10000, 'gtol': 1e-3},
    )
    runtime = _time.time() - t0
    est_r = res_opt.x
    print(est_r[:p])
    print(runtime)

    out_path = os.path.join(
        res_dir,
        f'res_Gtransform_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        est_r=est_r.reshape(-1, 1), runtime=runtime,
        k=k, l=l, p=p, q=q, knots=knots,
    )

    if ci:
        se_all = inference(N, seed, data_setting, knots_setting, root=root)
        out_path_se = os.path.join(
            res_dir,
            f'res_Gtransform_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        np.savez_compressed(
            out_path_se,
            est_r=est_r.reshape(-1, 1), se_all=se_all.reshape(-1, 1),
            runtime=runtime,
            k=k, l=l, p=p, q=q, knots=knots,
        )
    return est_r, runtime


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 3
    knots_setting = args[3] if len(args) > 3 else 'equal'
    ci = bool(int(args[4])) if len(args) > 4 else False
    main(N, seed, data_setting, knots_setting, ci)
