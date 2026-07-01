"""LTM smoke test on a small patient subsample.

The LTM model fixes ``beta_1 = 1`` for identifiability, so we put the
strongest Cox feature (bicarbonate_mean) first as anchor — Cox |z| there
is ~7.5, so the relative scale is well-defined.

Tries each knots setting (K1..K4) on a 500-patient subsample, ci=False,
to find a configuration where the iterative MLE converges (succ_ind=1).
"""
from __future__ import annotations

import os
import sys
import time as _time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

ROOT = '/Users/bomeng/Desktop/research/review/jmlr/code/mimic_analysis'
NPZ  = os.path.join(ROOT, 'merged_data/cohort_long_scaled.npz')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recurrent_ode import fit  # noqa: E402

ADM_DUMMIES = ['Adm_type_scheduled', 'Adm_type_unscheduled']


def load(n_patients=500, seed=0):
    npz = np.load(NPZ, allow_pickle=True)
    cols = [c.decode() if isinstance(c, bytes) else str(c)
            for c in npz['x_cols']]
    x = npz['x']
    id_ = npz['id'].astype(int)
    time = npz['time'].astype(float)
    delta = npz['delta'].astype(int)

    # one-hot Adm_type 0/1/2 -> two dummies
    adm_idx = cols.index('Adm_type')
    adm_col = x[:, adm_idx]
    sched = (adm_col == 1).astype(float)
    unsched = (adm_col == 2).astype(float)
    x = np.delete(x, adm_idx, axis=1)
    cols = [c for c in cols if c != 'Adm_type']
    x = np.concatenate([x, sched.reshape(-1, 1), unsched.reshape(-1, 1)],
                       axis=1)
    cols = cols + ADM_DUMMIES

    # Anchor: put bicarbonate_mean first (LTM fixes beta_1 = 1).
    anchor = 'bicarbonate_mean'
    j = cols.index(anchor)
    perm = [j] + [k for k in range(len(cols)) if k != j]
    x = x[:, perm]
    cols = [cols[k] for k in perm]

    rng = np.random.default_rng(seed)
    uniq = np.unique(id_)
    take = rng.choice(uniq, size=n_patients, replace=False)
    take = np.sort(take)
    mask = np.isin(id_, take)
    sub_id = id_[mask]
    # remap to consecutive 1..N
    mapping = {old: i + 1 for i, old in enumerate(np.unique(sub_id))}
    sub_id_new = np.array([mapping[v] for v in sub_id], dtype=int)
    return ({
        'x': x[mask], 'time': time[mask],
        'delta': delta[mask], 'id': sub_id_new,
    }, cols)


def main():
    data, cols = load(n_patients=500)
    print(f'subsample: rows={len(data["time"])} '
          f'patients={int(data["id"].max())} '
          f'features={data["x"].shape[1]} '
          f'events={int((data["delta"] == 1).sum())}')
    print(f'first feature (anchor with beta=1): {cols[0]}')

    for kn in ['K1', 'K2', 'K3', 'K4']:
        print(f'\n--- knots={kn} (ci=False) ---')
        t0 = _time.time()
        try:
            est = fit(data, model='ltm', random_effect=True,
                      knots=kn, ci=False)
            dt = _time.time() - t0
            print(f'  succ={est.success}  runtime={dt:.1f}s '
                  f'beta[:5]={est.beta[:5]}')
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
