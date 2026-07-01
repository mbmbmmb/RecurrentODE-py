"""Unified entry point for the RecurrentODE family of estimators.

Wraps the per-setting implementations under ``RecurrentODE_py`` behind a
single ``fit(data, model=..., random_effect=..., ...)`` call and returns
a uniform :class:`Estimate` with point estimates, sandwich SEs and 95%
Wald confidence intervals for the regression coefficients ``beta``.
"""
from .api import Estimate, fit, simulate

__all__ = ['Estimate', 'fit', 'simulate']
