"""AFT recurrent events + AFT-type survival censoring.

* Recurrent event intensity (AFT, setting 2):
      lambda(t|x) = 2 * exp(x'beta) / sqrt(4 * exp(x'beta) * t + 1),
  beta = (1, 1, 1).
* Censoring time C|x is drawn from an AFT-type survival model with the
  same intensity shape used by the AFT recurrent-event generator:
      h_C(t|x) = 2 * lambda_0_C * exp(x'gamma)
                 / sqrt(4 * lambda_0_C * exp(x'gamma) * t + 1)
  Cumulative hazard:
      H_C(t|x) = sqrt(4 * lambda_0_C * exp(x'gamma) * t + 1) - 1.
  Inversion: V ~ Exp(1) gives
      C = (V^2 + 2V) / (4 * lambda_0_C * exp(x'gamma)).
  Conditionally (given x) C and N(.) are independent; marginally
  correlated via x.
* Fixed upper bound TAU caps follow-up at min(C, TAU). Imperceptible
  per-subject jitter (<= 1e-6) is added only to subjects at the cap or
  below the numerical floor, so that the downstream aft.cox_rec
  solve_ivp sees a strictly-increasing t_eval.
"""
from __future__ import annotations

import os
import numpy as np

from ...common import ensure_dir


GAMMA = np.array([1.0, 1.0, 1.0])
LAMBDA_0_C = 0.2
TAU = 4.0
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
                  gamma=GAMMA, lambda_0_c=LAMBDA_0_C,
                  tau=TAU, u_min=U_MIN):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ensure_dir(data_dir)

    rng = np.random.default_rng(seed)

    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])

    rate_c = lambda_0_c * np.exp(x @ gamma)
    v_exp = -np.log(rng.random(N))           # V ~ Exp(1)
    C = (v_exp ** 2 + 2.0 * v_exp) / (4.0 * rate_c)

    u = np.minimum(C, tau)
    tie_noise = TIE_EPS * rng.random(N)
    u = np.where(u >= tau, tau + tie_noise, u)
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
    print(f'Admin-capped subjects (C >= tau): {int(np.sum(C >= tau))}/{N}')
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
