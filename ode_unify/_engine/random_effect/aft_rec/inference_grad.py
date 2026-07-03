"""Port of random effect/aft_rec/inference_grad.m.

The MATLAB source is a byte-for-byte duplicate of ``objective_func_grad``;
this module simply re-exports it.
"""
from __future__ import annotations

from .objective_func_grad import objective_func_grad as inference_grad

__all__ = ['inference_grad']
