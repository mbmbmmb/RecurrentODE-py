"""Port of ltm/main.m."""
from __future__ import annotations

import os
import time as _time
import numpy as np

from ..common import ensure_dir
from .generator_rec import generator_rec
from .mle import mle
from .inference import inference


def main(N, seed, data_setting, knots_setting, ci=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir = os.path.join(root, 'data')
    res_dir = os.path.join(root, 'res')
    ensure_dir(data_dir); ensure_dir(res_dir)
    print(seed)

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

    t0 = _time.time()
    est, knots_0, knots_q, k0, kq, q_0, q_q = mle(
        x, time, delta, id_vec, knots_setting,
    )
    runtime = _time.time() - t0
    est_r = est[:-1]
    succ_ind = int(est[-1])

    out_path = os.path.join(
        res_dir,
        f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        est_r=est_r.reshape(-1, 1), runtime=runtime,
        succ_ind=succ_ind,
        knots_0=knots_0, knots_q=knots_q,
        k0=k0, kq=kq, q_0=q_0, q_q=q_q,
    )

    if succ_ind and ci:
        se_all = inference(N, seed, data_setting, knots_setting, root=root)
        out_path_se = os.path.join(
            res_dir,
            f'res_ltm_N{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_se.npz',
        )
        np.savez_compressed(
            out_path_se,
            est_r=est_r.reshape(-1, 1), runtime=runtime,
            succ_ind=succ_ind, se_all=se_all.reshape(-1, 1),
            knots_0=knots_0, knots_q=knots_q,
            k0=k0, kq=kq, q_0=q_0, q_q=q_q,
        )
        print('Standard Error Calculated!')

    return est_r, runtime, succ_ind


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 4
    knots_setting = args[3] if len(args) > 3 else 'K4'
    ci = bool(int(args[4])) if len(args) > 4 else False
    main(N, seed, data_setting, knots_setting, ci)
