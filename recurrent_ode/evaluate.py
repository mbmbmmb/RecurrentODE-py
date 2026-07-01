"""Evaluate and visualize a fitted recurrent-event model.

Loads an estimate file produced by :mod:`recurrent_ode.estimate`,
reconstructs the functional parameters from the saved spline pieces,
prints a summary against the known truth, and (with ``--plot``) renders
a matplotlib figure with pointwise 95% confidence bands.

Plot semantics
--------------
- **cox / random-effect cox**: baseline hazard ``alpha(t)``.
- **aft / random-effect aft**: AFT-like sieve ``q(u)``.
- **npmle**: G-transformation rate ``q(u)``.
- **ltm / random-effect ltm**: two panels, ``alpha(t)`` and ``q(u)``,
  with the same identifiability rescaling used by the per-setting
  ``visual.py`` (so the curves overlay the truth one-to-one).

Examples
--------
::

    python -m recurrent_ode.evaluate \
        --estimate estimates/cox_n200_seed1.npz \
        --plot plots/cox_n200_seed1.png

    python -m recurrent_ode.evaluate \
        --estimate estimates/ltm_setting4.npz \
        --plot plots/ltm_setting4.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RecurrentODE_py.common import spcol


# True curves used by the per-setting ``visual.py`` modules. Single-spline
# models share one truth function per data_setting.

def _true_cox_alpha(t):
    # data_setting 1: lambda(t) = t^2 + 1
    return t ** 2 + 1.0


def _true_aft_q(u):
    # data_setting 2: q(u) = 2 / (1 + u)
    return 2.0 / (1.0 + u)


def _true_npmle_q(u):
    # data_setting 3: q(u) = 0.2 / (1 + u)
    return 0.2 / (1.0 + u)


def _ltm_truth(data_setting, pxt, pxq):
    """Returns (true_alpha, true_q, scale_a, scale_q) for ltm settings."""
    if data_setting == 1:
        return pxt ** 3, np.ones_like(pxq), 1.5 ** 3, 1.0 / 1.5 ** 3
    if data_setting == 2:
        return np.full_like(pxt, 2.0), np.exp(-pxq), 2.0, 0.5
    if data_setting == 3:
        return np.ones_like(pxt), 2.0 / (1.0 + pxq), 1.0, 1.0
    if data_setting == 4:
        return pxt + 1.0, 2.0 / (1.0 + pxq), 3.0, 1.0 / 3.0
    raise ValueError(f'unknown data_setting={data_setting}')


def _eval_single_spline(beta_len, est_r, se_arr, knots, k, grid):
    """Reconstruct ``exp(B(grid) @ theta)`` plus 95% pointwise CI.

    For the single-spline models (cox/aft/npmle), the layout is
    ``[beta(beta_len), theta(q)]`` in both ``est_r`` and ``se_all``.
    """
    theta = est_r[beta_len:]
    B = spcol(knots, k, np.asarray(grid, dtype=float).ravel())
    est = np.exp(B @ theta)
    if se_arr is not None and se_arr.size >= est_r.size:
        se_theta = se_arr[beta_len:]
        upper = np.exp(B @ (theta + 1.96 * se_theta))
        lower = np.exp(B @ (theta - 1.96 * se_theta))
    else:
        upper = lower = None
    return est, upper, lower


def _eval_ltm_splines(est_r, se_arr, raw, p, grid_t, grid_q):
    """Reconstruct (alpha, q) curves for the LTM family.

    ``est_r`` layout: [1.0, b2, ..., bp, theta(q_q), alpha(q_0)].
    ``se_arr`` layout: [se_b2, ..., se_bp, se_theta, se_alpha] — note no
    SE entry for the fixed first beta.
    """
    knots_0 = raw['knots_0'].ravel()
    knots_q = raw['knots_q'].ravel()
    k0 = int(raw['k0'].ravel()[0])
    kq = int(raw['kq'].ravel()[0])
    q_0 = int(raw['q_0'].ravel()[0])
    q_q = int(raw['q_q'].ravel()[0])

    theta = est_r[p:p + q_q]
    alpha = est_r[p + q_q:p + q_q + q_0]

    grid_t = np.asarray(grid_t, dtype=float).ravel()
    grid_q = np.asarray(grid_q, dtype=float).ravel()
    Bq = spcol(knots_q, kq, grid_q)
    B0 = spcol(knots_0, k0, grid_t)

    est_q = np.exp(Bq @ theta)
    est_a = np.exp(B0 @ alpha)

    if se_arr is not None and se_arr.size >= (p - 1) + q_q + q_0:
        se_theta = se_arr[p - 1:p - 1 + q_q]
        se_alpha = se_arr[p - 1 + q_q:p - 1 + q_q + q_0]
        est_q_upper = np.exp(Bq @ (theta + 1.96 * se_theta))
        est_q_lower = np.exp(Bq @ (theta - 1.96 * se_theta))
        est_a_upper = np.exp(B0 @ (alpha + 1.96 * se_alpha))
        est_a_lower = np.exp(B0 @ (alpha - 1.96 * se_alpha))
    else:
        est_q_upper = est_q_lower = est_a_upper = est_a_lower = None

    return (est_a, est_a_upper, est_a_lower,
            est_q, est_q_upper, est_q_lower,
            knots_0, knots_q)


def _meta(payload, key, default=None):
    if key not in payload:
        return default
    v = payload[key]
    if v.dtype.kind == 'b':
        return bool(v.item())
    if v.dtype.kind in 'iu':
        return int(v.item())
    return str(v.item())


def _summary(name, grid, est, true_y, lower, upper):
    err = est - true_y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    line = f'  {name}: bias={bias:+.4f} rmse={rmse:.4f}'
    if lower is not None:
        cov = float(np.mean((true_y >= lower) & (true_y <= upper)))
        line += f' point-cov95={cov:.2f}'
    print(line)


def _plot_single(grid, est, lower, upper, true_y, xlabel, ylabel,
                 title, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(grid, true_y, 'b', lw=2, label='True')
    plt.plot(grid, est, 'r', lw=2, label='Estimate')
    if lower is not None:
        plt.plot(grid, upper, '--', color=(0.929, 0.694, 0.125), lw=1)
        plt.plot(grid, lower, '--', color=(0.929, 0.694, 0.125), lw=1,
                 label='95% CI')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'  plot -> {out_path}')


def _plot_ltm(grid_t, est_a, lower_a, upper_a, true_a,
              grid_q, est_q, lower_q, upper_q, true_q,
              title, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grid_t, true_a, 'b', lw=2, label='True')
    axes[0].plot(grid_t, est_a, 'r', lw=2, label='Estimate')
    if lower_a is not None:
        axes[0].plot(grid_t, upper_a, '-.',
                     color=(0.929, 0.694, 0.125), lw=1)
        axes[0].plot(grid_t, lower_a, '-.',
                     color=(0.929, 0.694, 0.125), lw=1, label='95% CI')
    axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$\alpha(t)$')
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(grid_q, true_q, 'b', lw=2)
    axes[1].plot(grid_q, est_q, 'r', lw=2)
    if lower_q is not None:
        axes[1].plot(grid_q, upper_q, '-.',
                     color=(0.929, 0.694, 0.125), lw=1)
        axes[1].plot(grid_q, lower_q, '-.',
                     color=(0.929, 0.694, 0.125), lw=1)
    axes[1].set_xlabel('u'); axes[1].set_ylabel('q(u)')
    axes[1].grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f'  plot -> {out_path}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Evaluate / visualize a fitted recurrent-event model.',
    )
    p.add_argument('--estimate', required=True,
                   help='Path to a .npz produced by estimate.py.')
    p.add_argument('--data', default=None,
                   help='Optional simulated-data .npz, used to infer the '
                        'functional-parameter grid range.')
    p.add_argument('--plot', default=None,
                   help='Optional output path for the matplotlib figure.')
    p.add_argument('--n-grid', type=int, default=200)
    p.add_argument('--t-min', type=float, default=None)
    p.add_argument('--t-max', type=float, default=None)
    p.add_argument('--u-min', type=float, default=None)
    p.add_argument('--u-max', type=float, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw = np.load(args.estimate, allow_pickle=False)
    payload = {k: raw[k] for k in raw.files}

    model = _meta(payload, '_model')
    random_effect = _meta(payload, '_random_effect', False)
    data_setting = _meta(payload, '_data_setting', -1)
    if data_setting == -1:
        data_setting = None
    knots_setting = _meta(payload, '_knots_setting', '')

    print(f'model={model} random_effect={random_effect} '
          f'data_setting={data_setting} knots={knots_setting}')

    beta = payload['beta'].ravel()
    se = payload['se'].ravel() if 'se' in payload else None
    print(f'beta = {np.array2string(beta, precision=4)}')
    if se is not None:
        print(f'se   = {np.array2string(se, precision=4)}')
        for j in range(beta.size):
            if np.isnan(se[j]):
                continue
            lo = beta[j] - 1.96 * se[j]
            hi = beta[j] + 1.96 * se[j]
            print(f'  beta[{j}]: [{lo: .4f}, {hi: .4f}]')

    raw_keys = {k[len('raw_'):]: v for k, v in payload.items()
                if k.startswith('raw_')}
    est_r = raw_keys['est_r'].ravel()
    se_all = raw_keys.get('se_all')
    if se_all is not None:
        se_all = se_all.ravel()
    p_dim = beta.size

    # Pick functional-parameter grid ranges. Read the data file's time
    # column for sane defaults; fall back to [0, 4] otherwise.
    t_max = args.t_max
    u_max = args.u_max
    if (t_max is None or u_max is None) and args.data:
        d = np.load(args.data)
        tmax_data = float(np.max(d['time']))
        if t_max is None:
            t_max = tmax_data
        if u_max is None:
            u_max = tmax_data
    if t_max is None: t_max = 4.0
    if u_max is None: u_max = 4.0
    t_min = 0.0 if args.t_min is None else args.t_min
    u_min = 0.0 if args.u_min is None else args.u_min
    grid_t = np.linspace(t_min, t_max, args.n_grid)
    grid_q = np.linspace(u_min, u_max, args.n_grid)

    is_ltm = model == 'ltm'
    if is_ltm:
        if data_setting is None:
            raise ValueError(
                'data_setting must be known to evaluate the LTM truth',
            )
        (est_a, est_a_up, est_a_lo,
         est_q, est_q_up, est_q_lo,
         _, _) = _eval_ltm_splines(est_r, se_all, raw_keys, p_dim,
                                   grid_t, grid_q)
        true_a, true_q, scale_a, scale_q = _ltm_truth(
            data_setting, grid_t, grid_q,
        )
        # Apply the same identifiability rescaling as the per-setting
        # ``ltm/visual.py`` so estimate and truth are on the same scale.
        est_a *= scale_a
        est_q *= scale_q
        if est_a_up is not None:
            est_a_up *= scale_a; est_a_lo *= scale_a
            est_q_up *= scale_q; est_q_lo *= scale_q

        print('Functional-parameter summaries:')
        _summary('alpha(t)', grid_t, est_a, true_a, est_a_lo, est_a_up)
        _summary('q(u)',     grid_q, est_q, true_q, est_q_lo, est_q_up)

        if args.plot:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.plot)) or '.',
                exist_ok=True,
            )
            title = (f'LTM (random_effect={random_effect}) '
                     f'data_setting={data_setting}')
            _plot_ltm(grid_t, est_a, est_a_lo, est_a_up, true_a,
                      grid_q, est_q, est_q_lo, est_q_up, true_q,
                      title, args.plot)
        return

    # Single-spline models: use raw_knots / raw_k.
    knots = raw_keys['knots'].ravel()
    k = int(raw_keys['k'].ravel()[0])

    if model == 'cox':
        true_y = _true_cox_alpha(grid_t)
        grid = grid_t
        xlabel, ylabel = 't', r'$\alpha(t)$'
    elif model == 'aft':
        true_y = _true_aft_q(grid_q)
        grid = grid_q
        xlabel, ylabel = 'u', 'q(u)'
    elif model == 'npmle':
        true_y = _true_npmle_q(grid_q)
        grid = grid_q
        xlabel, ylabel = 'u', 'q(u)'
    else:
        raise ValueError(f'unsupported model={model}')

    est, upper, lower = _eval_single_spline(
        p_dim, est_r, se_all, knots, k, grid,
    )
    print('Functional-parameter summary:')
    _summary(ylabel, grid, est, true_y, lower, upper)

    if args.plot:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.plot)) or '.',
            exist_ok=True,
        )
        title = (f'{model} (random_effect={random_effect}) '
                 f'data_setting={data_setting}')
        _plot_single(grid, est, lower, upper, true_y,
                     xlabel, ylabel, title, args.plot)


if __name__ == '__main__':
    main()
