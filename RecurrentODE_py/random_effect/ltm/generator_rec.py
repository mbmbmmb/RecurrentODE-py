"""Port of random effect/ltm/generator_rec.m.

Same frailty-thinning logic as ``random_effect/cox``/``aft_rec``; the only
difference is the on-disk layout: MATLAB wrote to ``data/cox_rec_rd/`` or
``data/aft_rd/`` depending on ``data_setting``.
"""
from __future__ import annotations

import os

from ..cox.generator_rec import generator_rec as _generator_rec


def _data_subdir(data_setting):
    if data_setting == 1:
        return 'cox_rec_rd'
    if data_setting == 2:
        return 'aft_rd'
    raise ValueError(f'unsupported data_setting={data_setting}')


def generator_rec(N, seed, data_setting, rho1=0.0, r1=1.0, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), 'data', _data_subdir(data_setting),
        )
    return _generator_rec(N, seed, data_setting, rho1, r1, data_dir=data_dir)
