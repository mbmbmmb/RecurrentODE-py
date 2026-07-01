"""End-to-end example for the ``recurrent_ode`` package.

Builds a small long-format dataset by hand, fits the AFT-type recurrent
event model with gamma frailty and a quadratic B-spline on q(.), and
prints the regression coefficients with 95% Wald CIs.

Run as:
    python -m recurrent_ode.example
"""
from __future__ import annotations

import numpy as np

from recurrent_ode import fit, simulate


# --- Option A: bring your own data ----------------------------------------
# Long format: one row per event + one censoring row per subject.

def my_dataset():
    # 3 subjects, 1 covariate, 2 events for subj 1, 0 for subj 2, 1 for subj 3.
    rows = [
        # id, time, delta, x1
        (1, 0.4, 1, 0.7),
        (1, 1.1, 1, 0.7),
        (1, 2.0, 0, 0.7),   # censoring row
        (2, 1.5, 0, -0.3),  # no events at all -> just one row, delta=0
        (3, 0.9, 1, 0.1),
        (3, 2.5, 0, 0.1),
    ]
    arr = np.array(rows, dtype=float)
    return {
        'id':    arr[:, 0].astype(int),
        'time':  arr[:, 1],
        'delta': arr[:, 2].astype(int),
        'x':     arr[:, 3:],         # (n_rows, p=1)
    }


# --- Option B: simulate canonical synthetic data --------------------------

def main():
    # Simulate AFT-type data with N=300 subjects, p=3 covariates.
    data = simulate(N=300, seed=42, model='aft', data_setting=2)
    print('Generated long-format data:')
    print(f'  rows : {data["x"].shape[0]}')
    print(f'  subj : {len(np.unique(data["id"]))}')
    print(f'  cols : id, time, delta, x[{data["x"].shape[1]}]')
    print()

    # Fit the AFT-type estimator. ``ci=True`` adds sandwich SE and Wald CIs.
    est = fit(
        data,
        model='aft',          # 'cox' / 'aft' / 'npmle' / 'ltm'
        random_effect=False,  # set True for the gamma-frailty version
        knots='quantile',     # 'quantile' or 'equal' (AFT/NPMLE/LTM only)
        ci=True,
    )

    print('Estimates:')
    for j, b in enumerate(est.beta):
        if est.se is not None and not np.isnan(est.se[j]):
            print(f'  beta[{j}] = {b:+.4f} '
                  f'(se {est.se[j]:.4f}, '
                  f'95% CI [{est.ci_lower[j]:+.4f}, {est.ci_upper[j]:+.4f}])')
        else:
            print(f'  beta[{j}] = {b:+.4f} (fixed)')

    # The functional parameter q(.) is held in est.spline:
    print()
    print(f'spline coefs : {est.spline["coefs"].shape}')
    print(f'spline knots : {est.spline["knots"].shape}')
    print(f'spline order : k = {est.spline["k"]}')
    print(f'runtime      : {est.runtime:.2f} s')

    # To reconstruct q(u) on a grid:
    from RecurrentODE_py.common import spcol
    grid = np.linspace(0.0, 4.0, 5)
    B = spcol(est.spline['knots'], est.spline['k'], grid)
    q_hat = np.exp(B @ est.spline['coefs'])
    print(f'\nq(u) on grid {grid.tolist()}:')
    print('  ', np.array2string(q_hat, precision=4))


if __name__ == '__main__':
    main()
