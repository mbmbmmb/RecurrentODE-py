"""Unified ``fit`` / ``simulate`` API for the RecurrentODE estimators.

The wrapper writes the user's data to a temp directory in the format
expected by the per-setting ``main()`` functions in ``RecurrentODE_py``,
calls those mains with ``root=<tempdir>``, then reads the saved
``_se.npz`` (or ``_fish.npz`` for ``random_effect.ltm``) result file
back. This guarantees that ``fit(...)`` produces estimates that are
numerically identical to the existing per-setting pipelines.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Make sure the parent of ``RecurrentODE_py`` is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Per (model, random_effect) configuration. The "always_regen" flag
# patches around ``aft.main`` which (intentionally, for parity with
# MATLAB) regenerates the data file even if one is already present.
_REG = {
    ('cox', False): dict(
        mod='RecurrentODE_py.cox.main',
        gen='RecurrentODE_py.cox.generator_rec',
        default_setting=1, default_knots=None, takes_knots=False,
        res_template='res/res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        data_subdir='data',
        se_key='se_all', fixed_b1=False, p=3, always_regen=False,
    ),
    ('aft', False): dict(
        mod='RecurrentODE_py.aft.main',
        gen='RecurrentODE_py.aft.generator_rec',
        default_setting=2, default_knots='quantile', takes_knots=True,
        res_template='res/res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        data_subdir='data',
        se_key='se_all', fixed_b1=False, p=3, always_regen=True,
    ),
    ('npmle', False): dict(
        mod='RecurrentODE_py.npmle.main',
        gen='RecurrentODE_py.npmle.generator_rec',
        default_setting=3, default_knots='equal', takes_knots=True,
        res_template='res/res_Gtransform_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        data_subdir='data',
        se_key='se_all', fixed_b1=False, p=3, always_regen=False,
    ),
    ('ltm', False): dict(
        mod='RecurrentODE_py.ltm.main',
        gen='RecurrentODE_py.ltm.generator_rec',
        default_setting=4, default_knots='K4', takes_knots=True,
        res_template='res/res_ltm_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        data_subdir='data',
        se_key='se_all', fixed_b1=True, p=3, always_regen=False,
    ),
    ('cox', True): dict(
        mod='RecurrentODE_py.random_effect.cox.main',
        gen='RecurrentODE_py.random_effect.cox.generator_rec',
        default_setting=1, default_knots=None, takes_knots=False,
        res_template='res/res_cox_N{N}_seed{seed}_setting{setting}_se.npz',
        data_subdir='data',
        se_key='se_beta', fixed_b1=False, p=3, always_regen=False,
    ),
    ('aft', True): dict(
        mod='RecurrentODE_py.random_effect.aft_rec.main',
        gen='RecurrentODE_py.random_effect.aft_rec.generator_rec',
        default_setting=2, default_knots='quantile', takes_knots=True,
        res_template='res/res_aft_N{N}_seed{seed}_setting{setting}_knots{knots}_se.npz',
        data_subdir='data',
        se_key='se_all', fixed_b1=False, p=3, always_regen=False,
    ),
    ('ltm', True): dict(
        mod='RecurrentODE_py.random_effect.ltm.main',
        gen='RecurrentODE_py.random_effect.ltm.generator_rec',
        default_setting=1, default_knots='K4', takes_knots=True,
        # Setting-specific subdirs; resolved at call time.
        res_template=None, data_subdir=None,
        se_key='se_all', fixed_b1=True, p=3, always_regen=False,
    ),
}


@dataclass
class Estimate:
    """Unified output from ``fit()``.

    Attributes
    ----------
    beta : np.ndarray
        Length-``p`` coefficient vector. For models that fix ``beta_1 = 1``
        for identifiability (LTM, random-effect LTM), index 0 is set to
        ``1.0``.
    se : np.ndarray | None
        Same length as ``beta``; ``NaN`` at fixed components.
    ci_lower / ci_upper : np.ndarray | None
        95% Wald CI: ``beta +/- 1.96 * se``. ``NaN`` at fixed components.
    spline : dict
        Functional-parameter spline data (``coefs``, ``knots``, ``k``).
    runtime : float
        Estimation time (seconds), excluding SE calculation.
    model : str
        ``"cox"`` / ``"aft"`` / ``"npmle"`` / ``"ltm"``.
    random_effect : bool
    success : bool
        ``False`` only for LTM where the constrained optimization failed.
    raw : dict
        Unfiltered contents of the underlying ``_se.npz`` (or ``_fish.npz``)
        file, useful for debugging or reproducing the per-setting pipeline.
    """
    beta: np.ndarray
    se: Optional[np.ndarray]
    ci_lower: Optional[np.ndarray]
    ci_upper: Optional[np.ndarray]
    spline: dict
    runtime: float
    model: str
    random_effect: bool
    success: bool = True
    raw: dict = field(default_factory=dict)


def _save_data_npz(data, path, model):
    payload = {
        'x': np.asarray(data['x']),
        'time': np.asarray(data['time']).reshape(-1, 1),
        'delta': np.asarray(data['delta']).reshape(-1, 1),
        'id': np.asarray(data['id']).reshape(-1, 1),
    }
    if model == 'npmle':
        payload['rho1'] = float(data.get('rho1', 0.5))
        payload['r1'] = float(data.get('r1', 1.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def _re_ltm_paths(root, N, seed, data_setting, knots, ci=True):
    sub = 'cox_rec_rd' if data_setting == 1 else 'aft_rd'
    prefix = 'res_cox_N' if data_setting == 1 else 'res_aft_N'
    data_path = os.path.join(
        root, 'data', sub,
        f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
    )
    suffix = '_fish.npz' if ci else '.npz'
    res_path = os.path.join(
        root, 'res', sub,
        f'{prefix}{N}_seed{seed}_setting{data_setting}_knots{knots}{suffix}',
    )
    return data_path, res_path


def _call_main(mod, key, N, seed, setting, knots, ci, root):
    if key in {('cox', False), ('cox', True)}:
        return mod.main(N, seed, setting, ci, root=root)
    if key == ('npmle', False):
        return mod.main(N, seed, setting, knots, ci, root=root)
    # aft / ltm / random_effect.aft / random_effect.ltm
    return mod.main(N, seed, setting, knots, ci, root=root)


def fit(data, *, model, random_effect=False, knots=None, data_setting=None,
        ci=True, seed=0):
    """Fit a recurrent-event model with optional gamma frailty.

    Parameters
    ----------
    data : dict
        Mapping with keys ``'x'`` (2D array, ``(n_rows, p)``), ``'time'``,
        ``'delta'`` (event indicator: 1 event, 0 censoring), and ``'id'``
        (subject identifier per row). For ``model='npmle'`` also
        ``'rho1'``, ``'r1'`` (defaults 0.5 / 1.0).
    model : {'cox', 'aft', 'npmle', 'ltm'}
    random_effect : bool, default False
        If True, fit the gamma-frailty version. ``model='npmle'`` is not
        supported with ``random_effect=True``.
    knots : str | None
        Required for ``aft``/``npmle``/``ltm``: ``'quantile'``, ``'equal'``,
        or ``'K1'..'K4'`` for ``ltm``.
    data_setting : int | None
        Underlying data-generating regime (1=Cox, 2=AFT, 3=Gtransform,
        4=Flex). Defaults to the canonical value for ``model``.
    ci : bool, default True
        Compute sandwich SE and 95% Wald CI for ``beta``.
    seed : int, default 0
        Synthetic seed used only in temp filenames.
    """
    key = (model, bool(random_effect))
    if key not in _REG:
        raise ValueError(
            f'unsupported (model={model!r}, random_effect={random_effect})'
        )
    cfg = _REG[key]

    if data_setting is None:
        data_setting = cfg['default_setting']
    if cfg['takes_knots']:
        if knots is None:
            knots = cfg['default_knots']
    else:
        knots = ''

    # N must be the *subject* count, which is what the per-setting mains and
    # filename templates expect; data['x'] is in long format so its row count
    # is generally larger.
    N = int(np.unique(np.asarray(data['id']).ravel()).size)
    seed = int(seed)

    with tempfile.TemporaryDirectory(prefix='recurrent_ode_') as root:
        if key == ('ltm', True):
            data_path, res_path = _re_ltm_paths(
                root, N, seed, data_setting, knots, ci=ci,
            )
        else:
            data_path = os.path.join(
                root, cfg['data_subdir'],
                f'simudata_N{N}_seed{seed}_setting{data_setting}.npz',
            )
            res_path = os.path.join(root, cfg['res_template'].format(
                N=N, seed=seed, setting=data_setting, knots=knots,
            ))

        _save_data_npz(data, data_path, model)

        mod = importlib.import_module(cfg['mod'])
        if cfg['always_regen']:
            # Patch out generator_rec for the duration of the call so our
            # supplied data file isn't overwritten (needed for AFT, whose
            # main always regenerates).
            original = mod.generator_rec
            mod.generator_rec = lambda *a, **kw: None
            try:
                _call_main(mod, key, N, seed, data_setting, knots, ci, root)
            finally:
                mod.generator_rec = original
        else:
            _call_main(mod, key, N, seed, data_setting, knots, ci, root)

        if not os.path.isfile(res_path):
            raise RuntimeError(
                f'expected result file not produced: {res_path}'
            )
        return _build_estimate(res_path, key, cfg, ci)


def _build_estimate(res_path, key, cfg, ci):
    res = np.load(res_path)
    files = set(res.files)
    est_r = np.asarray(res['est_r']).ravel()
    runtime = (
        float(np.asarray(res['runtime']).ravel()[0])
        if 'runtime' in files else 0.0
    )

    # The per-setting mains save the actual feature count `p` they fit
    # (= x.shape[1]). Read it back rather than relying on cfg's default,
    # which was wired to the simulator's p=3 toy size.
    if 'p' in files:
        p = int(np.asarray(res['p']).ravel()[0])
    else:
        p = cfg['p']
    fixed = cfg['fixed_b1']

    if fixed:
        # est_r layout: [1.0, b2, ..., bp, spline coefs]. The leading 1.0 is
        # already in est_r (mle prepends it for identifiability).
        beta_full = est_r[:p].copy()
        beta_full[0] = 1.0  # be explicit about the constraint
        spline_coefs = est_r[p:]
    else:
        beta_full = est_r[:p]
        spline_coefs = est_r[p:]

    se_full = ci_lower = ci_upper = None
    if ci and cfg['se_key'] in files:
        se_all = np.asarray(res[cfg['se_key']]).ravel()
        if fixed:
            se_full = np.concatenate([[np.nan], se_all[: p - 1]])
        else:
            se_full = se_all[:p]
        ci_lower = beta_full - 1.96 * se_full
        ci_upper = beta_full + 1.96 * se_full

    spline = {
        'coefs': spline_coefs,
        'knots': (
            np.asarray(res['knots']).ravel() if 'knots' in files else None
        ),
        'k': (
            int(np.asarray(res['k']).ravel()[0]) if 'k' in files else None
        ),
    }

    success = True
    if 'succ_ind' in files:
        success = bool(int(np.asarray(res['succ_ind']).ravel()[0]))

    raw = {k: np.asarray(res[k]) for k in files}
    return Estimate(
        beta=beta_full, se=se_full,
        ci_lower=ci_lower, ci_upper=ci_upper,
        spline=spline, runtime=runtime,
        model=key[0], random_effect=key[1],
        success=success, raw=raw,
    )


def simulate(N, seed, model, *, random_effect=False, data_setting=None):
    """Generate canonical simulation data via the existing ``generator_rec``.

    Returns a dict with the same keys as the corresponding ``simudata_*.npz``
    file (``x``, ``time``, ``delta``, ``id``, plus ``rho1``/``r1`` for
    ``model='npmle'``).
    """
    key = (model, bool(random_effect))
    if key not in _REG:
        raise ValueError(
            f'unsupported (model={model!r}, random_effect={random_effect})'
        )
    cfg = _REG[key]
    if data_setting is None:
        data_setting = cfg['default_setting']

    with tempfile.TemporaryDirectory(prefix='recurrent_ode_sim_') as td:
        gen_mod = importlib.import_module(cfg['gen'])
        path = gen_mod.generator_rec(N, seed, data_setting, data_dir=td)
        data = dict(np.load(path))
    return {k: np.asarray(v) for k, v in data.items()}
