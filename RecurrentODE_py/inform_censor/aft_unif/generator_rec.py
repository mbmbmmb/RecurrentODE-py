"""AFT recurrent events + Uniform(a(x), b(x)) censoring.

* Recurrent event intensity (AFT, setting 2):
      lambda(t|x) = 2 * exp(x'beta) / sqrt(4 * exp(x'beta) * t + 1),
  beta = (1, 1, 1).
* Censoring time C|x ~ Uniform(a(x), b(x)) with simple linear bounds:
      a(x) = A_MIN  (constant)
      b(x) = B_INTERCEPT - B_SLOPE * (x' gamma),  gamma = (1, 1, 1).
  Conditionally on x, C is independent of the recurrent event process.
* x' gamma in [-2, 3] gives b(x) in [2.5, 5], always > a(x). No admin
  cap (C is naturally bounded). U_MIN floor + tiny tie jitter only.
"""
from __future__ import annotations

import os
import numpy as np

from ...common import ensure_dir


GAMMA = np.array([1.0, 1.0, 1.0])
A_MIN = 0.2
B_INTERCEPT = 4.0
B_SLOPE = 0.5
U_MIN = 0.05
TIE_EPS = 1e-6

BETA = np.array([1.0, 1.0, 1.0])


def _intensity(data_setting):
    if data_setting == 1:
        return lambda t, xi, b: np.exp(xi @ b) * (t ** 2 + 1)
    if data_setting == 2:
        return lambda t, xi, b: 2 * np.exp(xi @ b) / np.sqrt(
            4 * np.exp(xi @ b) * t + 1
        )
    raise ValueError(f'unknown data_setting={data_setting}')


def generator_rec(N, seed, data_setting, data_dir=None,
                  gamma=GAMMA, a_min=A_MIN,
                  b_intercept=B_INTERCEPT, b_slope=B_SLOPE,
                  u_min=U_MIN):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ensure_dir(data_dir)

    rng = np.random.default_rng(seed)

    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])

    a_x = np.full(N, a_min)
    b_x = b_intercept - b_slope * (x @ gamma)
    b_x = np.maximum(b_x, a_x + 0.5)

    C = a_x + (b_x - a_x) * rng.random(N)

    u = C.copy()
    tie_noise = TIE_EPS * rng.random(N)
    u = np.where(u < u_min, u_min + tie_noise, u)

    intensity = _intensity(data_setting)

    rows = []
    for i in range(N):
        subj_rng = np.random.default_rng(seed * (i + 1))
        xi = x[i]
        rho = lambda t: intensity(t, xi, BETA)

        t_grid = np.linspace(0.0, u[i], 200)
        lambda_max = max(float(np.max(rho(t_grid))), 1e-9)

        t_events = []
        s = 0.0
        while s < u[i]:
            s = s - np.log(subj_rng.random()) / lambda_max
            if s < u[i]:
                if subj_rng.random() <= rho(s) / lambda_max:
                    t_events.append(s)

        n = len(t_events)
        if n == 0:
            rows.append(np.concatenate([[i + 1, u[i], 0.0], xi]))
        elif t_events[-1] < u[i]:
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

    print('Censoring Rate (%):', (1 - np.mean(delta)) * 100)
    print(f'C range: [{C.min():.3f}, {C.max():.3f}]  '
          f'b(x) range: [{b_x.min():.3f}, {b_x.max():.3f}]')
    print('Time Quantiles (33%, 66%):', np.quantile(time, [0.33, 0.66]))

    out_path = os.path.join(
        data_dir,
        f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    np.savez_compressed(
        out_path,
        x=x_out, time=time.reshape(-1, 1),
        delta=delta.reshape(-1, 1), id=id_col.reshape(-1, 1),
    )
    return out_path


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    N = int(args[0]) if len(args) > 0 else 200
    seed = int(args[1]) if len(args) > 1 else 1
    data_setting = int(args[2]) if len(args) > 2 else 2
    generator_rec(N, seed, data_setting)
