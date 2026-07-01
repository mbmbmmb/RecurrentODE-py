# RecurrentODE

A Python package for **recurrent event analysis with ordinary differential equations**, implementing the estimation and inference procedures from:

> [Recurrent Event Analysis with Ordinary Differential Equations](https://arxiv.org/abs/2507.20396)
> Bo Meng, Weijing Tang, Gongjun Xu, and Ji Zhu.

The rate function of a subject's recurrent-event process is modeled as an ODE

$$\mu'_x(t) = q\big(\mu(t)\big)\,\exp(x^\top\beta)\,\alpha(t),$$

and the package estimates the regression coefficients $\beta$ together with the
functional parameters $\alpha(\cdot)$ (baseline rate) and $q(\cdot)$ (transformation),
which are approximated by B-splines. Standard errors and 95% Wald confidence
intervals for $\beta$ are provided via a sandwich variance estimator.

---

## The models

Different choices of $q$ and $\alpha$ give the four model types, all fit through the same API:

| `model`  | Rate equation                                   | What is estimated              |
|----------|-------------------------------------------------|--------------------------------|
| `cox`    | $\mu'_x(t)=\exp(x^\top\beta)\,\alpha(t)$         | $\beta,\ \alpha(\cdot)$ (with $q\equiv1$) |
| `aft`    | $\mu'_x(t)=q(\mu(t))\,\exp(x^\top\beta)$         | $\beta,\ q(\cdot)$ (with $\alpha\equiv1$) |
| `npmle`  | $\mu'_x(t)=q(\mu(t)\mid\rho)\,\exp(x^\top\beta)\,\alpha(t)$ | $\beta,\ \alpha(\cdot)$ ($q$ specified — G-transformation) |
| `ltm`    | $\mu'_x(t)=q(\mu(t))\,\exp(x^\top\beta)\,\alpha(t)$ | $\beta,\ q(\cdot),\ \alpha(\cdot)$ (fully flexible) |

**Gamma-frailty (random effect).** Pass `random_effect=True` to add a mean-1 Gamma
random effect $\xi_i$ per subject, $\ \mu'_{x_i}(t)=\xi_i\,q(\mu(t))\,\exp(x_i^\top\beta)\,\alpha(t)$.
Supported for `cox`, `aft`, and `ltm`. *(Not available for `npmle`.)*

---

## Installation

Requires Python 3.9+ and:

```bash
pip install numpy scipy         # core
pip install matplotlib          # optional: plotting / confidence bands
```

No build step — clone the repo and run from its root so the package is importable.

---

## Quickstart

### 1. Bring your own data

Data is in **long format**: one row per observed event, plus one censoring row per
subject. Provide a dict with four arrays:

| key     | shape          | meaning                                  |
|---------|----------------|------------------------------------------|
| `id`    | `(n_rows,)`    | subject identifier for each row          |
| `time`  | `(n_rows,)`    | event / censoring time                   |
| `delta` | `(n_rows,)`    | 1 = event, 0 = censoring                  |
| `x`     | `(n_rows, p)`  | covariates ($p$ columns)                 |

```python
import numpy as np
from recurrent_ode import fit

# 3 subjects, 1 covariate. Subject 1 has 2 events then a censoring row;
# subject 2 has no events; subject 3 has 1 event then censoring.
data = {
    'id':    np.array([1, 1, 1, 2, 3, 3]),
    'time':  np.array([0.4, 1.1, 2.0, 1.5, 0.9, 2.5]),
    'delta': np.array([1,   1,   0,   0,   1,   0]),   # 0 rows are censoring
    'x':     np.array([[0.7],[0.7],[0.7],[-0.3],[0.1],[0.1]]),
}

est = fit(data, model='aft', knots='quantile', ci=True)

print(est.beta)       # coefficient vector, length p
print(est.se)         # sandwich standard errors (NaN at fixed components)
print(est.ci_lower, est.ci_upper)   # 95% Wald CI
```

> The 6-row example above is only to show the **data layout**. Meaningful
> coefficients and standard errors require a realistic number of subjects — with a
> handful of subjects the variance matrix is singular and `se` returns `NaN`.

### 2. Fit the gamma-frailty version

```python
est = fit(data, model='ltm', random_effect=True, knots='K4', ci=True)
```

### 3. Simulate canonical data, then fit

```python
from recurrent_ode import simulate, fit

data = simulate(N=300, seed=42, model='aft', data_setting=2)   # 300 subjects
est  = fit(data, model='aft', knots='quantile', ci=True)
```

A full runnable script is in [`recurrent_ode/example.py`](recurrent_ode/example.py):

```bash
python -m recurrent_ode.example
```

---

## The `fit` result

`fit(...)` returns an `Estimate` with:

| attribute              | description                                                  |
|------------------------|--------------------------------------------------------------|
| `beta`                 | length-$p$ coefficient vector (index 0 fixed to `1.0` for `ltm`) |
| `se`                   | sandwich SEs (same length; `NaN` at fixed components)        |
| `ci_lower` / `ci_upper`| 95% Wald CI, `beta ± 1.96·se`                                |
| `spline`               | `{'coefs', 'knots', 'k'}` — reconstruct $q(\cdot)$ / $\alpha(\cdot)$ |
| `runtime`              | estimation time in seconds                                   |
| `success`              | `False` only if the constrained `ltm` optimization failed    |
| `raw`                  | full contents of the underlying result file (for debugging)  |

Reconstruct a functional parameter on a grid:

```python
from RecurrentODE_py.common import spcol
grid  = np.linspace(0, 4, 50)
B     = spcol(est.spline['knots'], est.spline['k'], grid)
q_hat = np.exp(B @ est.spline['coefs'])
```

**Knot placement** (for `aft` / `npmle` / `ltm`): `'quantile'` places interior knots at
event-time quantiles; `'equal'` spaces them evenly. For `ltm`, use `'K1'`…`'K4'` to
control the number of knots.

---

## Command-line workflow

The package also ships three small CLIs for a simulate → estimate → evaluate pipeline:

```bash
# 1. Generate synthetic data
python -m recurrent_ode.simulate_data \
    --model cox --N 200 --seed 1 --out data/cox.npz

# 2. Fit and save estimates (coefficients, SE, CI, spline pieces)
python -m recurrent_ode.estimate \
    --data data/cox.npz --out estimates/cox.npz

# 3. Print a summary and render 95% confidence bands
python -m recurrent_ode.evaluate \
    --estimate estimates/cox.npz --plot plots/cox.png
```

---

## Repository layout

```
recurrent_ode/       Unified API — start here
  api.py               fit() / simulate() / Estimate
  estimate.py          CLI: fit a saved dataset
  simulate_data.py     CLI: generate synthetic data
  evaluate.py          CLI: reconstruct functionals + plot CI bands
  example.py           end-to-end example

RecurrentODE_py/     Reference implementations (the "engine room")
  cox/  aft/  npmle/  ltm/     per-model estimators used in the paper
  random_effect/              gamma-frailty variants (cox, aft_rec, ltm)
  inform_censor/              informative-censoring stress tests
  test/                       coverage / reproducibility studies
  README.md                   detailed per-model documentation

mimic_analysis/      MIMIC-III real-data application (see note below)
```

The `recurrent_ode` wrapper is the recommended interface; the `RecurrentODE_py`
subpackages are the exact per-model pipelines behind it and are documented in
[`RecurrentODE_py/README.md`](RecurrentODE_py/README.md). The original MATLAB
implementation lives separately at
[github.com/mbmbmmb/RecurrentODE](https://github.com/mbmbmmb/RecurrentODE).

### A note on the MIMIC data

The `mimic_analysis/` scripts apply the method to the **MIMIC-III** clinical database.
MIMIC-III is credentialed, proprietary data and is **not** redistributed in this
repository — neither the raw data nor patient-level derived files are included. To
reproduce that analysis you must obtain MIMIC-III access
([physionet.org](https://physionet.org/content/mimiciii/)) and regenerate the inputs
locally. Only aggregate result plots are provided.

---

## Citation

```bibtex
@article{meng2025recurrent,
  title  = {Recurrent Event Analysis with Ordinary Differential Equations},
  author = {Meng, Bo and Tang, Weijing and Xu, Gongjun and Zhu, Ji},
  journal = {arXiv preprint arXiv:2507.20396},
  year   = {2025}
}
```

## Contact

Bo Meng (University of Michigan) — bomeng@umich.edu
