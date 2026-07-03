"""Port of random effect/ltm/main.m.

Setting-specific on-disk layout (``data/cox_rec_rd/``, ``res/cox_rec_rd/``
for setting 1; ``data/aft_rd/``, ``res/aft_rd/`` for setting 2) matches
MATLAB so ``summary_cox`` / ``summary_aft`` can re-use the same .mat files.
"""
from __future__ import annotations

import os
import time as _time
import numpy as np

from ...common import ensure_dir
from .generator_rec import generator_rec
from .mle import mle
from .inference import inference


def _paths(root, data_setting):
    if data_setting == 1:
        data_sub, res_sub, res_prefix = 'cox_rec_rd', 'cox_rec_rd', 'res_cox_N'
    elif data_setting == 2:
        data_sub, res_sub, res_prefix = 'aft_rd', 'aft_rd', 'res_aft_N'
    else:
        raise ValueError(f'unsupported data_setting={data_setting}')
    data_dir = os.path.join(root, 'data', data_sub)
    res_dir = os.path.join(root, 'res', res_sub)
    return data_dir, res_dir, res_prefix


def main(N, seed, data_setting, knots_setting, ci=False, root=None):
    if root is None:
        root = os.path.dirname(__file__)
    data_dir, res_dir, res_prefix = _paths(root, data_setting)
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

    t0 = _time.time()
    est, p, q_q, q_0, knots_0, knots_q, k0, kq, l0, lq = mle(
        x, time, delta, knots_setting,
    )
    runtime = _time.time() - t0
    est_r = est[:-1]
    succ_ind = int(est[-1])

    out_path = os.path.join(
        res_dir,
        f'{res_prefix}{N}_seed{seed}_setting{data_setting}'
        f'_knots{knots_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        est_r=est_r.reshape(-1, 1), runtime=runtime,
        succ_ind=succ_ind,
        p=p, q_q=q_q, q_0=q_0,
        knots_0=knots_0, knots_q=knots_q,
        k0=k0, kq=kq, l0=l0, lq=lq,
    )

    if ci:
        fish = inference(N, seed, data_setting, knots_setting, root=root)
        se_all = np.sqrt(np.abs(np.diag(fish)))
        print(se_all)
        out_path_fish = os.path.join(
            res_dir,
            f'{res_prefix}{N}_seed{seed}_setting{data_setting}'
            f'_knots{knots_setting}_fish.npz',
        )
        np.savez_compressed(
            out_path_fish,
            fish=fish, se_all=se_all.reshape(-1, 1),
            est_r=est_r.reshape(-1, 1), runtime=runtime,
            succ_ind=succ_ind,
            p=p, q_q=q_q, q_0=q_0,
            knots_0=knots_0, knots_q=knots_q,
            k0=k0, kq=kq, l0=l0, lq=lq,
        )
    return est_r, runtime, succ_ind


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 1
    knots_setting = args[3] if len(args) > 3 else 'K4'
    ci = bool(int(args[4])) if len(args) > 4 else False
    main(N, seed, data_setting, knots_setting, ci)
