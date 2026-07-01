"""Simulate recurrent-event data and save to ``.npz``.

A thin CLI around :func:`recurrent_ode.simulate` that produces canonical
synthetic data for any (model, random_effect, data_setting) combination
supported by the unified API. The resulting file has long-format columns
``x``, ``time``, ``delta``, ``id`` (and ``rho1``/``r1`` for the NPMLE
G-transformation generator) and is consumable by :mod:`estimate`.

Examples
--------
::

    python -m recurrent_ode.simulate_data \
        --model cox --N 200 --seed 1 \
        --out data/cox_n200_seed1.npz

    python -m recurrent_ode.simulate_data \
        --model ltm --random-effect --data-setting 2 \
        --N 500 --seed 7 --out data/re_ltm_aft.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import simulate
from .api import _REG


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Generate canonical recurrent-event simulation data.',
    )
    p.add_argument('--model', required=True,
                   choices=['cox', 'aft', 'npmle', 'ltm'])
    p.add_argument('--random-effect', action='store_true',
                   help='Use the gamma-frailty version of the generator.')
    p.add_argument('--N', type=int, required=True,
                   help='Number of subjects.')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data-setting', type=int, default=None,
                   help='1=Cox, 2=AFT, 3=Gtransform, 4=Flex (defaults to '
                        'the canonical value for the chosen model).')
    p.add_argument('--out', required=True,
                   help='Output .npz path.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = _REG[(args.model, bool(args.random_effect))]
    resolved_setting = (
        args.data_setting if args.data_setting is not None
        else cfg['default_setting']
    )
    data = simulate(
        args.N, args.seed, args.model,
        random_effect=args.random_effect,
        data_setting=resolved_setting,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {k: np.asarray(v) for k, v in data.items()}
    payload['_model'] = np.array(args.model)
    payload['_random_effect'] = np.array(bool(args.random_effect))
    payload['_data_setting'] = np.array(resolved_setting)
    payload['_N'] = np.array(args.N)
    payload['_seed'] = np.array(args.seed)
    np.savez_compressed(args.out, **payload)

    n_subj = int(np.unique(payload['id'].ravel()).size)
    n_rows = int(payload['x'].shape[0])
    n_evt = int(payload['delta'].sum())
    print(f'wrote {args.out}: {n_subj} subjects, {n_rows} rows, '
          f'{n_evt} events ({100 * n_evt / n_rows:.1f}%)')


if __name__ == '__main__':
    main()
