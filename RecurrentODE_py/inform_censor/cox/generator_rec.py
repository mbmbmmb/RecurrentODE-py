"""Generator for the ``inform_censor`` Cox recurrent-event setting.

Differences vs the standard ``cox/generator_rec.py`` setting 1:

* Recurrent event intensity is unchanged: lambda(t|x) = (t^2 + 1) * exp(x'beta),
  beta = (1, 1, 1).
* Censoring time is drawn from a **Cox survival** model conditional on x:
      C | x ~ Exp( lambda_0_C * exp(x' gamma) )
  i.e. h_C(t|x) = lambda_0_C * exp(x' gamma) with constant baseline. This
  gives conditional (given x) independence between C and the recurrent
  event process, but marginal correlation -- the exact "covariate-dependent
  non-informative censoring" setting.
* A fixed upper bound TAU caps the follow-up at ``min(C, TAU)`` so the
  thinning loop terminates in bounded time. An imperceptible per-subject
  jitter (<= 1e-6, ~6 orders below any real event time) is added to
  subjects sitting at the cap / below the numerical floor so that the
  downstream ``aft.cox_rec`` ODE solve (which requires strictly-increasing
  t_eval) does not see ties. No per-subject admin-cap randomisation
  beyond that.

The on-disk format matches ``cox/generator_rec.py`` byte-for-byte so the
existing ``cox.objective_func``, ``cox.inference`` and summary machinery
work without changes.
"""
from __future__ import annotations

import os
import numpy as np

from ...common import ensure_dir


# gamma of similar magnitude to beta so the marginal dependence between C
# and the recurrent event count is strong; lambda_0_C tuned so that for
# the "average" subject (x=0) the median of C is ~ 2.3.
GAMMA = np.array([1.0, 1.0, 1.0])
LAMBDA_0_C = 0.3
TAU = 4.0        # fixed upper bound on follow-up
U_MIN = 0.05     # numerical floor; protects solve_ivp at t=1e-12 for
                 # extreme-x subjects with near-zero C|x.
TIE_EPS = 1e-6   # jitter used ONLY to break ties at the cap / floor.

BETA = np.array([1.0, 1.0, 1.0])


def _true_hazard_ode(data_setting):
    """Same list of intensities as cox/generator_rec.py."""
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

    # Covariates -- identical to cox/setting 1.
    x1 = rng.standard_normal(N); x1 = np.clip(x1, -1, 1)
    x2 = rng.standard_normal(N); x2 = np.clip(x2, -1, 1)
    x3 = (rng.random(N) < 0.5).astype(float)
    x = np.column_stack([x1, x2, x3])

    # Cox-survival censoring times: C | x ~ Exp(rate_i).
    rate_c = lambda_0_c * np.exp(x @ gamma)
    u_exp = rng.random(N)
    C = -np.log(u_exp) / rate_c
    # u = min(C, tau), then enforce a numerical floor u_min. Ties at the
    # two endpoints get imperceptible jitter (<= 1e-6) so solve_ivp sees a
    # strictly-increasing t_eval downstream. C itself is continuous so
    # non-boundary u values are already distinct almost surely.
    u = np.minimum(C, tau)
    tie_noise = TIE_EPS * rng.random(N)
    u = np.where(u >= tau, tau + tie_noise, u)
    u = np.where(u < u_min, u_min + tie_noise, u)

    intensity = _true_hazard_ode(data_setting)

    rows = []
    for i in range(N):
        subj_rng = np.random.default_rng(seed * (i + 1))
        xi = x[i]
        rho = lambda t: intensity(t, xi, BETA)

        num_points = 200
        t_grid = np.linspace(0.0, u[i], num_points)
        lambda_max = float(np.max(rho(t_grid)))
        if lambda_max < 1e-9:
            lambda_max = 1e-9

        t_events = []
        s = 0.0
        while s < u[i]:
            s = s - np.log(subj_rng.random()) / lambda_max
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

    print('Censoring Rate (%):')
    print((1 - np.mean(delta)) * 100)
    print('Admin-capped subjects (C >= tau):  '
          f'{int(np.sum(C >= tau))}/{N}')
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
