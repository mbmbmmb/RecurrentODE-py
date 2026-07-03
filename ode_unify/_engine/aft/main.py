"""Port of aft/main.m."""
from __future__ import annotations

import os
import numpy as np

from ..common import augknt, ensure_dir
from .generator_rec import generator_rec
from .cox_rec import cox_rec
from .mle import mle
from .inference import inference


def main(N, seed, data_setting, knots_setting='quantile', ci=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir = os.path.join(root, 'data')
    res_dir = os.path.join(root, 'res')
    ensure_dir(data_dir); ensure_dir(res_dir)

    data_file = os.path.join(
        data_dir, f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    # MATLAB always regenerates; we keep that behaviour for parity.
    generator_rec(N, seed, data_setting, data_dir=data_dir)
    data = np.load(data_file)
    x = data['x']
    time = data['time'].ravel()
    delta = data['delta'].ravel()
    id_vec = data['id'].ravel()

    temp, beta = cox_rec(x, time, delta)

    k = 4
    l = int(np.ceil(N ** (1.0 / 5.0))) + 1

    if knots_setting == 'quantile':
        interior = np.quantile(temp, np.arange(1, l) / l)
        breaks = np.concatenate([[0.0], interior, [float(np.max(temp))]])
    elif knots_setting == 'equal':
        breaks = np.linspace(0.0, 2.0 * float(np.max(temp)), l + 1)
    else:
        raise ValueError(f'unknown knots_setting={knots_setting}')
    knots = augknt(breaks, k)

    p = x.shape[1]
    q = len(knots) - k

    forward = True
    r0 = np.zeros(p + q)
    r0[:p] = beta

    est_r, runtime = mle(x, time, delta, id_vec, knots, k, r0, forward)
    print(runtime)

    out_path = os.path.join(
        res_dir,
        f'res_aft_N{N}_seed{seed}_setting{data_setting}_knots{knots_setting}.npz',
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
            f'res_aft_N{N}_seed{seed}_setting{data_setting}_knots{knots_setting}_se.npz',
        )
        np.savez_compressed(
            out_path_se,
            est_r=est_r.reshape(-1, 1), runtime=runtime,
            se_all=se_all.reshape(-1, 1),
            k=k, l=l, p=p, q=q, knots=knots,
        )

    return est_r, runtime


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 2
    knots_setting = args[3] if len(args) > 3 else 'quantile'
    ci = bool(int(args[4])) if len(args) > 4 else False
    est, rt = main(N, seed, data_setting, knots_setting, ci)
    print('est:', est)
    print('runtime:', rt)
