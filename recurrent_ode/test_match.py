"""Verify the unified ``recurrent_ode.fit`` matches the per-setting ``main()``.

For each supported (model, random_effect) combination this:

  1. Generates canonical data via ``simulate(N, seed, model=...)``.
  2. Writes that data into a fresh temp directory in the layout each
     per-setting ``main()`` expects, then calls ``main()`` directly with
     ``root=<tempdir>`` to capture baseline ``est_r`` and the SE array.
  3. Calls ``recurrent_ode.fit(data, ...)`` on the same data.
  4. Asserts the unified ``Estimate`` matches the baseline file
     element-for-element (up to ``atol=1e-12``).

Because ``fit`` internally calls the same ``main()``, the two results
ought to be byte-identical -- this test is a wiring check.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recurrent_ode import fit, simulate
from recurrent_ode.api import _REG, _re_ltm_paths, _save_data_npz, _call_main


CASES = [
    # ---- canonical per-setting pipelines (one per module) ----
    dict(model='cox',   random_effect=False, knots=None,        data_setting=1),
    dict(model='aft',   random_effect=False, knots='quantile',  data_setting=2),
    dict(model='aft',   random_effect=False, knots='equal',     data_setting=2),
    dict(model='npmle', random_effect=False, knots='equal',     data_setting=3),
    dict(model='npmle', random_effect=False, knots='quantile',  data_setting=3),
    # ---- LTM is the flexible estimator: applied to all four data settings ----
    dict(model='ltm',   random_effect=False, knots='K4',        data_setting=1),
    dict(model='ltm',   random_effect=False, knots='K4',        data_setting=2),
    dict(model='ltm',   random_effect=False, knots='K4',        data_setting=3),
    dict(model='ltm',   random_effect=False, knots='K4',        data_setting=4),
    # ---- random-effect (gamma frailty) pipelines ----
    dict(model='cox',   random_effect=True,  knots=None,        data_setting=1),
    dict(model='aft',   random_effect=True,  knots='quantile',  data_setting=2),
    dict(model='aft',   random_effect=True,  knots='equal',     data_setting=2),
    # RE-LTM applied to both Cox-type and AFT-type frailty data
    dict(model='ltm',   random_effect=True,  knots='K4',        data_setting=1),
    dict(model='ltm',   random_effect=True,  knots='K4',        data_setting=2),
]

N = 200
SEED = 1
ATOL = 1e-12


def _baseline(case, data):
    """Run the per-setting ``main()`` with *data* placed into a temp root.

    Returns ``(est_r, se_arr_or_None)`` from the saved result file.
    """
    key = (case['model'], bool(case['random_effect']))
    cfg = _REG[key]
    knots = case['knots'] if cfg['takes_knots'] else ''
    ds = case['data_setting']

    with tempfile.TemporaryDirectory(prefix='baseline_') as root:
        if key == ('ltm', True):
            data_path, res_path = _re_ltm_paths(root, N, SEED, ds, knots)
        else:
            data_path = os.path.join(
                root, cfg['data_subdir'],
                f'simudata_N{N}_seed{SEED}_setting{ds}.npz',
            )
            res_path = os.path.join(root, cfg['res_template'].format(
                N=N, seed=SEED, setting=ds, knots=knots,
            ))
        _save_data_npz(data, data_path, case['model'])

        mod = importlib.import_module(cfg['mod'])
        if cfg['always_regen']:
            original = mod.generator_rec
            mod.generator_rec = lambda *a, **kw: None
            try:
                _call_main(mod, key, N, SEED, ds, knots, True, root)
            finally:
                mod.generator_rec = original
        else:
            _call_main(mod, key, N, SEED, ds, knots, True, root)

        if not os.path.isfile(res_path):
            raise RuntimeError(f'baseline did not produce {res_path}')
        res = np.load(res_path)
        est_r = np.asarray(res['est_r']).ravel().copy()
        se_key = cfg['se_key']
        se_arr = (
            np.asarray(res[se_key]).ravel().copy()
            if se_key in res.files else None
        )
    return est_r, se_arr


def _check(case):
    print(f'--- {case} ---')
    data = simulate(
        N, SEED, case['model'],
        random_effect=case['random_effect'],
        data_setting=case['data_setting'],
    )
    base_est_r, base_se = _baseline(case, data)

    out = fit(
        data,
        model=case['model'],
        random_effect=case['random_effect'],
        knots=case['knots'],
        data_setting=case['data_setting'],
        ci=True,
        seed=SEED,
    )

    cfg = _REG[(case['model'], bool(case['random_effect']))]
    p = cfg['p']
    if cfg['fixed_b1']:
        expect_beta = base_est_r[:p].copy()
        expect_beta[0] = 1.0
    else:
        expect_beta = base_est_r[:p]
    if not np.allclose(out.beta, expect_beta, atol=ATOL, rtol=0):
        raise AssertionError(
            f'beta mismatch: unified={out.beta} baseline={expect_beta}'
        )

    if base_se is None:
        if out.se is not None:
            raise AssertionError('unified produced se but baseline did not')
    else:
        if cfg['fixed_b1']:
            expect_se = np.concatenate([[np.nan], base_se[: p - 1]])
        else:
            expect_se = base_se[:p]
        ok = np.allclose(
            out.se, expect_se, atol=ATOL, rtol=0, equal_nan=True,
        )
        if not ok:
            raise AssertionError(
                f'se mismatch: unified={out.se} baseline={expect_se}'
            )

    if out.ci_lower is not None:
        lo = out.beta - 1.96 * out.se
        hi = out.beta + 1.96 * out.se
        assert np.allclose(out.ci_lower, lo, equal_nan=True)
        assert np.allclose(out.ci_upper, hi, equal_nan=True)

    print(f'  beta = {out.beta}')
    if out.se is not None:
        print(f'  se   = {out.se}')
    print('  OK')


def main():
    for case in CASES:
        _check(case)
    print('\nAll cases match the per-setting pipelines.')


if __name__ == '__main__':
    main()
