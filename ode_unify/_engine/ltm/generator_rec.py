"""Port of ltm/generator_rec.m."""
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
    u = 2 + 2 * rng.random(N)

    if data_setting == 1:
        true_hazard_ode = lambda t, xi, b: np.exp(xi @ b) * (t ** 2 + 1)
    elif data_setting == 2:
        true_hazard_ode = lambda t, xi, b: 2 * np.exp(xi @ b) / np.sqrt(4 * np.exp(xi @ b) * t + 1)
    elif data_setting == 3:
        alpha0 = 0.2
        true_hazard_ode = lambda t, xi, b: (
            np.exp(xi @ b) * (alpha0 / (1 + t))
            * (1 + np.exp(xi @ b) * alpha0 * np.log1p(t)) ** (rho1 - 1)
        )
    elif data_setting == 4:
        true_hazard_ode = lambda t, xi, b: (
            2 * np.exp(xi @ b) * (t + 1)
            / np.sqrt(2 * np.exp(xi @ b) * t * (t + 2) + 1)
        )
    else:
        raise ValueError(f'unknown data_setting={data_setting}')

    rows = []
    for i in range(N):
        subj_rng = np.random.default_rng(seed * (i + 1))
        xi = x[i]
        rho = lambda t: true_hazard_ode(t, xi, beta)
        t_grid = np.linspace(0, u[i], 200)
        lambda_max = max(float(np.max(rho(t_grid))), 1e-9)

        t_events = []
        s = 0.0
        while s < u[i]:
            s = s - np.log(subj_rng.random()) / lambda_max
            if s < u[i] and subj_rng.random() <= rho(s) / lambda_max:
                t_events.append(s)

        n = len(t_events)
        if n == 0:
            rows.append(np.concatenate([[i + 1, u[i], 0.0], xi]))
        else:
            if t_events[-1] < u[i]:
                for j in range(n):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xi]))
                rows.append(np.concatenate([[i + 1, u[i], 0.0], xi]))
            else:
                t_events[-1] = u[i]
                for j in range(n - 1):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xi]))
                rows.append(np.concatenate([[i + 1, t_events[-1], 0.0], xi]))

    out = np.array(rows)
    id_col = out[:, 0].astype(int)
    time = out[:, 1]
    delta = out[:, 2]
    x_out = out[:, 3:]

    print(np.quantile(time, [0.33, 0.66]))
    print(1 - delta.mean())

    out_path = os.path.join(
        data_dir, f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        x=x_out, time=time.reshape(-1, 1),
        delta=delta.reshape(-1, 1), id=id_col.reshape(-1, 1),
        rho1=rho1, r1=r1,
    )
    return out_path


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if args else 200
    seed = int(args[1]) if len(args) > 1 else 1
    ds = int(args[2]) if len(args) > 2 else 4
    generator_rec(N, seed, ds)
