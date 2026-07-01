"""Port of npmle/baseline/summary.m."""
from __future__ import annotations

import os
import numpy as np

from .main import main as run_main


def summary(N=1000, rep=1000, p=3, root=None, run=True):
    if root is None:
        root = os.path.dirname(__file__)
    res = np.zeros((rep, 8))
    se_res = np.zeros((rep, 8))
    runtimes = np.zeros(rep)

    for seed in range(1, rep + 1):
        print(seed)
        if run:
            run_main(N, seed, root=root)
        out_file = os.path.join(
            root, 'res_generator',
            f'npmle_N{N}_seed{seed}_setting_NPMLE.npz',
        )
        if not os.path.isfile(out_file):
            continue
        out = np.load(out_file)
        res[seed - 1, :] = out['beta_hat'].ravel()[:8]
        se_res[seed - 1, :] = out['se_beta'].ravel()[:8]
        runtimes[seed - 1] = float(out['runtime'].ravel()[0])

    est_beta = res[:, :p]
    se_beta = se_res[:, :p]
    true_beta = np.array([1.0, 1.0, 1.0])
    print('beta bias:'); print(est_beta.mean(0) - true_beta)
    print(est_beta.std(0, ddof=1))
    print('ESE:'); print(se_beta.mean(0))
    up = est_beta + 1.96 * se_beta
    low = est_beta - 1.96 * se_beta
    cp = ((true_beta < up) & (true_beta > low)).astype(float)
    print('CP:'); print(cp.mean(0))
    print('NPMLE n=1000 rep=1000 no rd effs')

    table = np.vstack([
        est_beta.mean(0) - true_beta, est_beta.std(0, ddof=1),
        se_beta.mean(0), cp.mean(0),
    ])
    print(table)
    return table


if __name__ == '__main__':
    summary()
