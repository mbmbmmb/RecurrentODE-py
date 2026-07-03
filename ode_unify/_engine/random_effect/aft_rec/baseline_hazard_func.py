"""Port of random effect/aft_rec/baseline_hazard_func.m (identical to aft/)."""
from __future__ import annotations

import numpy as np

from ...common import spcol


def baseline_hazard_func(time, theta, knots, k):
    time = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    B = spcol(knots, k, time)
    return np.exp(B @ theta)
