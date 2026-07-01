"""Fit a recurrent-event model from a simulated ``.npz`` file.

Loads data produced by :mod:`recurrent_ode.simulate_data`, calls the
unified :func:`recurrent_ode.fit`, and writes a self-contained estimate
file (regression coefficients, sandwich SE, 95 % Wald CI, plus the raw
spline pieces needed by :mod:`recurrent_ode.evaluate` to reconstruct the
functional parameter(s)).

Examples
--------
::

    python -m recurrent_ode.estimate \
        --data data/cox_n200_seed1.npz \
        --out  estimates/cox_n200_seed1.npz

    python -m recurrent_ode.estimate \
        --data data/re_ltm_aft.npz \
        --knots K4 \
        --out  estimates/re_ltm_aft.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import fit


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Fit a recurrent-event model from simulated data.',
    )
    p.add_argument('--data', required=True,
                   help='Path to a .npz file produced by simulate_data.py.')
    p.add_argument('--out', required=True,
                   help='Output .npz path for the estimate.')
    p.add_argument('--model', default=None,
                   choices=['cox', 'aft', 'npmle', 'ltm'],
                   help='Override the model recorded in the data file.')
    p.add_argument('--random-effect', dest='random_effect',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Override: use (or do not use) the gamma-frailty '
                        'estimator. Default: use the data file metadata.')
    p.add_argument('--knots', default=None,
                   help='Knot scheme for AFT/NPMLE/LTM '
                        '(e.g. quantile, equal, K1..K4).')
    p.add_argument('--data-setting', type=int, default=None)
    p.add_argument('--no-ci', action='store_true',
                   help='Skip sandwich SE and Wald CI.')
    return p.parse_args(argv)


def _meta_from_data(payload, key, fallback=None):
    if key in payload:
        v = payload[key]
        if v.dtype.kind == 'b':
            return bool(v.item())
        if v.dtype.kind in 'iu':
            return int(v.item())
        return str(v.item())
    return fallback


def main(argv=None):
    args = parse_args(argv)

    raw = np.load(args.data, allow_pickle=False)
    payload = {k: raw[k] for k in raw.files}
    data = {k: payload[k] for k in ('x', 'time', 'delta', 'id')}
    if 'rho1' in payload:
        data['rho1'] = float(payload['rho1'])
    if 'r1' in payload:
        data['r1'] = float(payload['r1'])

    model = args.model or _meta_from_data(payload, '_model', 'cox')
    if args.random_effect is not None:
        random_effect = bool(args.random_effect)
    else:
        random_effect = _meta_from_data(payload, '_random_effect', False)
    if args.data_setting is not None:
        data_setting = args.data_setting
    else:
        ds = _meta_from_data(payload, '_data_setting', -1)
        data_setting = None if ds == -1 else int(ds)
    seed = _meta_from_data(payload, '_seed', 0)

    est = fit(
        data, model=model, random_effect=random_effect,
        knots=args.knots, data_setting=data_setting,
        ci=not args.no_ci, seed=int(seed) if seed is not None else 0,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save = {
        'beta': est.beta,
        'success': np.array(est.success),
        'runtime': np.array(est.runtime),
        '_model': np.array(model),
        '_random_effect': np.array(bool(random_effect)),
        '_data_setting': np.array(
            data_setting if data_setting is not None else -1,
        ),
        '_knots_setting': np.array(args.knots if args.knots else ''),
    }
    if est.se is not None:
        save['se'] = est.se
        save['ci_lower'] = est.ci_lower
        save['ci_upper'] = est.ci_upper
    # Stash the raw spline machinery so evaluate.py can reconstruct
    # alpha(t) / q(u) without re-running the estimator.
    for k, v in est.raw.items():
        save[f'raw_{k}'] = np.asarray(v)
    np.savez_compressed(args.out, **save)

    print(f'model={model} random_effect={random_effect} '
          f'data_setting={data_setting} knots={args.knots}')
    print(f'beta = {np.array2string(est.beta, precision=4)}')
    if est.se is not None:
        print(f'se   = {np.array2string(est.se, precision=4)}')
        print('95% Wald CI:')
        for j, (lo, hi) in enumerate(zip(est.ci_lower, est.ci_upper)):
            print(f'  beta[{j}]: [{lo: .4f}, {hi: .4f}]')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
