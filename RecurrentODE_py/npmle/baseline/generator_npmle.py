"""Port of npmle/baseline/generator_npmle.m.

The NPMLE baseline uses a fixed observation horizon ``tau = 4`` (no random
censoring) and the Box-Cox intensity with ``rho1=0.5, r1=1``.  Files are
saved under ``data_generator/simudata_N<N>_seed<seed>_setting_<label>.mat``
in the current ``baseline`` directory.
"""
from __future__ import annotations

import os
import numpy as np

from ...common import ensure_dir


def generator_npmle(N, seed, data_setting, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data_generator')
    ensure_dir(data_dir)

    rho1 = 0.5
    r1 = 1.0
    beta = np.array([1.0, 1.0, 1.0])
    alpha0 = 0.2
    tau = 4.0

    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])

    def true_hazard_ode(t, xi, b):
        m = np.exp(xi @ b)
        return (m * (alpha0 / (1.0 + t))
                * (1.0 + m * alpha0 * np.log1p(t)) ** (rho1 - 1.0))

    u = np.full(N, tau)
    results = []
    for i in range(1, N + 1):
        sub_rng = np.random.default_rng(seed * i)
        xi = x[i - 1]
        t_grid = np.linspace(0.0, u[i - 1], 500)
        hazard_grid = true_hazard_ode(t_grid, xi, beta)
        lambda_max = float(np.max(hazard_grid)) * 1.1
        if lambda_max < 1e-9:
            lambda_max = 1e-9

        t_events = []
        s = 0.0
        while s < u[i - 1]:
            s = s - np.log(sub_rng.random()) / lambda_max
            if s < u[i - 1]:
                if sub_rng.random() <= true_hazard_ode(s, xi, beta) / lambda_max:
                    t_events.append(s)

        if not t_events:
            block = np.column_stack([[i], [u[i - 1]], [0.0], xi[None, :]])
        else:
            t_events = np.asarray(t_events)
            n = t_events.size
            times_i = np.concatenate([t_events, [u[i - 1]]])
            delta_i = np.concatenate([np.ones(n), [0.0]])
            ids = np.full(n + 1, i)
            xs = np.broadcast_to(xi, (n + 1, xi.size))
            block = np.column_stack([ids, times_i, delta_i, xs])
        results.append(block)

    out = np.vstack(results)
    ID = out[:, 0].astype(int)
    Y = out[:, 1]
    Delta = out[:, 2]
    X = out[:, 3:]

    print('Simulation finished. Apparent censoring rate:')
    print(1.0 - float(np.mean(Delta)))

    suffix = '' if rho1 > 0 else '_log_trans'
    out_path = os.path.join(
        data_dir,
        f'simudata_N{N}_seed{seed}_setting_{data_setting}{suffix}.npz',
    )
    np.savez_compressed(
        out_path,
        X=X, Y=Y.reshape(-1, 1), Delta=Delta.reshape(-1, 1),
        ID=ID.reshape(-1, 1), rho1=rho1, r1=r1,
    )
    return out_path
