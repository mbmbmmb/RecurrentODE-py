"""Port of npmle/generator_rec.m.

Simulates recurrent-event data under the G-transformation model.  The data
settings encode the true intensity function (Cox / AFT / Box-Cox
transformation / a hybrid).  Output ``.mat`` files are saved alongside the
MATLAB originals but use numpy's PCG64 RNG so they will not match MATLAB
bit-for-bit.
"""
from __future__ import annotations

import os
import numpy as np

from ..common import ensure_dir


def generator_rec(N, seed, data_setting, rho1=0.5, r1=1.0, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ensure_dir(data_dir)

    beta = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed)

    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])
    u = 2.0 + 2.0 * rng.random(N)

    if data_setting == 1:
        def true_hazard_ode(t, xi, b):
            return np.exp(xi @ b) * (t ** 2 + 1)
    elif data_setting == 2:
        def true_hazard_ode(t, xi, b):
            m = np.exp(xi @ b)
            return 2.0 * m / np.sqrt(4.0 * m * t + 1.0)
    elif data_setting == 3:
        alpha0 = 0.2

        def true_hazard_ode(t, xi, b):
            m = np.exp(xi @ b)
            return (m * (alpha0 / (1.0 + t))
                    * (1.0 + m * alpha0 * np.log1p(t)) ** (rho1 - 1.0))
    elif data_setting == 4:
        def true_hazard_ode(t, xi, b):
            m = np.exp(xi @ b)
            return (2.0 * m * (t + 1.0)
                    / np.sqrt(2.0 * m * t * (t + 2.0) + 1.0))
    else:
        raise ValueError(f'unknown data_setting={data_setting}')

    results = []
    for i in range(1, N + 1):
        sub_rng = np.random.default_rng(seed * i)
        xi = x[i - 1]
        t_grid = np.linspace(0.0, u[i - 1], 200)
        hazard_grid = true_hazard_ode(t_grid, xi, beta)
        lambda_max = float(np.max(hazard_grid))
        if lambda_max < 1e-9:
            lambda_max = 1e-9

        t_events = []
        s = 0.0
        while s < u[i - 1]:
            s = s - np.log(sub_rng.random()) / lambda_max
            if s < u[i - 1]:
                if sub_rng.random() <= true_hazard_ode(s, xi, beta) / lambda_max:
                    t_events.append(s)

        n = len(t_events)
        if n == 0:
            block = np.column_stack([[i], [u[i - 1]], [0.0], xi[None, :]])
        else:
            t_events = np.asarray(t_events)
            if t_events[-1] < u[i - 1]:
                times_i = np.concatenate([t_events, [u[i - 1]]])
                delta_i = np.concatenate([np.ones(n), [0.0]])
                m = n + 1
            else:
                t_events[-1] = u[i - 1]
                times_i = t_events
                delta_i = np.concatenate([np.ones(n - 1), [0.0]])
                m = n
            ids = np.full(m, i)
            xs = np.broadcast_to(xi, (m, xi.size))
            block = np.column_stack([ids, times_i, delta_i, xs])
        results.append(block)

    out = np.vstack(results)
    id_vec = out[:, 0].astype(int)
    time = out[:, 1]
    delta = out[:, 2]
    x_out = out[:, 3:]

    print(np.quantile(time, [0.33, 0.66]))
    print(1.0 - float(np.mean(delta)))

    data_path = os.path.join(
        data_dir,
        f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        data_path,
        x=x_out, time=time.reshape(-1, 1), delta=delta.reshape(-1, 1),
        id=id_vec.reshape(-1, 1), rho1=rho1, r1=r1,
    )
    return data_path
