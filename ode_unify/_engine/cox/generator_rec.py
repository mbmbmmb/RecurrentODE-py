"""Port of cox/generator_rec.m.

Generates recurrent-event data under four different intensity models via the
thinning algorithm and saves the result to ``data/simudata_N*_seed*_setting*.mat``.

Note: Python's RNG (PCG64 by default in numpy >= 2) does not match MATLAB's
Mersenne Twister stream, so numerical values will differ from the MATLAB run
with the same seed, but the simulation logic and model definitions are
faithful ports.
"""
from __future__ import annotations

import os
import numpy as np

from ..common import ensure_dir


def generator_rec(N, seed, data_setting, rho1=0.5, r1=1.0, data_dir=None):
    """Simulate N recurrent-event trajectories and write them to a .mat file."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ensure_dir(data_dir)

    beta = np.array([1.0, 1.0, 1.0])

    rng = np.random.default_rng(seed)

    # Two truncated-normal covariates and one binary covariate.
    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])

    # Administrative censoring time ~ U(2, 4).
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

        num_points = 200
        t_grid = np.linspace(0.0, u[i], num_points)
        lambda_max = float(np.max(rho(t_grid)))
        if lambda_max < 1e-9:
            lambda_max = 1e-9

        n = 0
        t_events = []
        s = 0.0
        while s < u[i]:
            s = s - np.log(subj_rng.random()) / lambda_max
            if subj_rng.random() <= rho(s) / lambda_max:
                t_events.append(s)
                n += 1

        if not t_events:
            rows.append(np.concatenate([[i + 1, u[i], 0.0], xi]))
        else:
            if t_events[-1] < u[i]:
                # Last event observed; append a censoring row at u[i].
                for j in range(n):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xi]))
                rows.append(np.concatenate([[i + 1, u[i], 0.0], xi]))
            else:
                # Last proposed event was past u[i]; convert it into the censoring record.
                t_events[-1] = u[i]
                for j in range(n - 1):
                    rows.append(np.concatenate([[i + 1, t_events[j], 1.0], xi]))
                rows.append(np.concatenate([[i + 1, t_events[-1], 0.0], xi]))

    out = np.array(rows)
    id_col = out[:, 0].astype(int)
    time = out[:, 1]
    delta = out[:, 2]
    x_out = out[:, 3:]

    print('Censoring Rate (%):')
    print((1 - np.mean(delta)) * 100)
    print('Time Quantiles (33%, 66%):')
    print(np.quantile(time, [0.33, 0.66]))

    out_path = os.path.join(
        data_dir,
        f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        x=x_out,
        time=time.reshape(-1, 1),
        delta=delta.reshape(-1, 1),
        id=id_col.reshape(-1, 1),
    )
    return out_path


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if len(args) > 0 else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 1
    generator_rec(N, seed, data_setting)
