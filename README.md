# RecurrentODE

A Python implementation of **recurrent event analysis with ordinary differential equations**, from:

> [Recurrent Event Analysis with Ordinary Differential Equations](https://arxiv.org/abs/2507.20396)
> Bo Meng, Weijing Tang, Gongjun Xu, and Ji Zhu.

Each subject's recurrent-event mean function solves the ODE

$$\mu'_x(t) = \xi\,q\big(\mu(t)\big)\,\exp(x^\top\beta)\,\alpha(t),\qquad \mu(0)=0,$$

with an optional subject-level frailty $\xi$. The methods estimate the regression
coefficients $\beta$ together with the functional parameters $\alpha(\cdot)$
(baseline rate) and $q(\cdot)$ (transformation), approximated by B-splines, and
provide standard errors and 95% Wald confidence intervals.

---

## The models

Different choices of $q$ and $\alpha$ give the four model types:

| `estimator` | Rate equation                                   | What is estimated              |
|-------------|-------------------------------------------------|--------------------------------|
| `cox`       | $\mu'_x(t)=\exp(x^\top\beta)\,\alpha(t)$         | $\beta,\ \alpha(\cdot)$ (with $q\equiv1$) |
| `aft`       | $\mu'_x(t)=q(\mu(t))\,\exp(x^\top\beta)$         | $\beta,\ q(\cdot)$ (with $\alpha\equiv1$) |
| `npmle`     | $\mu'_x(t)=q(\mu(t)\mid\rho)\,\exp(x^\top\beta)\,\alpha(t)$ | $\beta,\ \alpha(\cdot)$ ($q$ specified — G-transformation) |
| `ltm`       | $\mu'_x(t)=q(\mu(t))\,\exp(x^\top\beta)\,\alpha(t)$ | $\beta,\ q(\cdot),\ \alpha(\cdot)$ (fully flexible) |

**Frailty (random effect).** `random_effect=True` adds a subject multiplier
$\xi_i$ (mean-1 Gamma by default): $\ \mu'_{x_i}(t)=\xi_i\,q(\mu(t))\,\exp(x_i^\top\beta)\,\alpha(t)$.
Supported for `cox`, `aft`, and `ltm`.

---

## Installation

Requires Python 3.9+ and:

```bash
pip install numpy scipy         # core
pip install matplotlib          # plots
```

No build step — clone the repo and run from its root so the packages are importable.

---

## Repository layout

```
ode_unify/             ★ The unified package — one module per pipeline stage
  dgp.py                 simulate(): one data generator for every setting/frailty
  estimator.py           estimate(): fast point estimation (no SEs)
  inference.py           inference(): standard errors; fit() = estimate + inference
  visual.py              curve() / plot_fit() / band_plot() / ltm_band_plot()
  simulate_study.py      Monte-Carlo study driver + CLI (reproduces the paper figures)
  sanity_check.py        exact-parity checks vs the per-model reference code
  _engine/               vendored numerical kernels (objectives, MLE, inference)
  plots/                 the 9 simulation band plots

mimic_analysis/        MIMIC-III real-data application (data not included; see note)

recurrent_ode/         Legacy thin wrapper (superseded by ode_unify)
recurrent_ode_unify/   Earlier consolidation step (superseded by ode_unify)
RecurrentODE_py/       Legacy per-model reference implementations
```

`ode_unify` is the recommended interface. Its numerics are **verified identical**
to the per-model reference implementations (see
[Reproducibility](#reproducibility)). The original MATLAB code lives at
[github.com/mbmbmmb/RecurrentODE](https://github.com/mbmbmmb/RecurrentODE).

---

## 1. Data generation — `ode_unify.simulate`

One generator covers the whole family: the intensity is chosen by a preset
`setting`, by custom functional parameters `alpha(t)` / `q(u)` (integrated
through the ODE), or by a closed-form `rate(t, m)` with `m = exp(x'beta)`; the
frailty is any distribution, or off.

```python
import numpy as np
import ode_unify

# presets 1-4 (bit-identical to the paper's generators)
data = ode_unify.simulate(1000, seed=1, setting=1)                     # Cox
data = ode_unify.simulate(1000, seed=1, setting=2, random_effect=True) # AFT + gamma frailty

# custom alpha(t), q(u):  intensity from  mu' = m q(mu) alpha(t)
ode_unify.simulate(1000, seed=1, alpha=lambda t: t + 1.0,
                   q=lambda u: 2.0 / (1.0 + u))                        # == setting 4

# custom closed-form rate, custom frailty distribution, custom beta
ode_unify.simulate(1000, seed=1, rate=lambda t, m: m * np.exp(-t))
ode_unify.simulate(1000, seed=1, setting=1, random_effect=True,
                   frailty_dist='lognormal', frailty_params=(0.5,))
ode_unify.simulate(1000, seed=1, setting=1, beta=(1.0, 0.5))           # 2 covariates
```

Returns **long-format** data: one row per event plus one censoring row per
subject — `id`, `time`, `delta` (1 event / 0 censoring), `x` (n_rows × p).

---

## 2. Estimation — `ode_unify.estimate`

Point estimation is deliberately **separate from inference**, because it is
fast (seconds) while standard errors can be much slower (the frailty
resampling runs 50–800 perturbed score evaluations).

```python
est = ode_unify.estimate(data, estimator='ltm', random_effect=True, knots='K4')
est.beta            # regression coefficients (beta_1 pinned to 1 for ltm)
est.spline          # fitted B-spline pieces for alpha(.) / q(.)
est.se              # None — no inference yet
```

**Knot schemes:** `'quantile'` / `'equal'` for `aft` and `npmle`;
`'K1'`…`'K4'` for `ltm` (defaults: aft `quantile`, npmle `equal`, ltm `K4`).

---

## 3. Inference — `ode_unify.inference`

Fills in SEs and 95% Wald CIs on a fitted `Estimate`. The method is chosen
automatically from the model:

| model | β covariance | spline SEs |
|---|---|---|
| cox / aft / npmle / ltm (no frailty) | closed form (empirical-Fisher inversion) | closed form |
| cox + frailty | **closed-form adjustment** | resampling (B=50), computed when `spline_se=True` |
| aft / ltm + frailty | resampling automatically (B=150 / 800) | same resampling |

```python
est = ode_unify.inference(est, data)        # adds est.se, est.ci_lower/upper, est.se_all
# or one call:
est = ode_unify.fit(data, estimator='cox', ci=True)
```

`layout='uniform'` (default) uses one memory layout everywhere so all
`ode_unify` results are mutually consistent; `layout='legacy'` reproduces each
old per-model pipeline **bit-for-bit** (the two agree to optimizer tolerance,
~1e-6, far below the SEs).

---

## 4. Visualization — `ode_unify.visual`

```python
from ode_unify import plot_fit, band_plot

# one fitted replication vs the truth
plot_fit(est, 'fit_cox.png', grid=np.linspace(0.1, 2.5, 100),
         truth=lambda t: t**2 + 1)

# many replications of the same setting
band_plot(list_of_estimates, 'band_cox.png',
          truth=lambda t: t**2 + 1, grid=np.linspace(0.1, 2.5, 60))
```

### How the 95% band plots are constructed

Each figure in [`ode_unify/plots/`](ode_unify/plots/) is built in two stages
from **100 independent replications** per setting (N = 1000 subjects; 2000 for
npmle):

**Stage 1 — pointwise Wald band, per replication.** A fitted functional
parameter is a log-linear B-spline, e.g. $\hat\alpha(t)=\exp(B(t)^\top\hat\theta)$
with basis $B(t)$ on a 60-point grid. Using the spline-coefficient standard
errors $se_\theta$ from the inference step, the band is formed on the log scale
and exponentiated:

$$\exp\!\big(B(t)^\top(\hat\theta \pm 1.96\,se_\theta)\big).$$

This is a pointwise 95% confidence band (1.96 = normal 97.5% quantile).

**Stage 2 — aggregation across replications.** The red line is the pointwise
**mean of the 100 estimate curves**; the dashed lines are the **mean of the 100
upper bands** and the **mean of the 100 lower bands** (an *average Wald band*,
not a percentile envelope). The frailty-LTM figure uses the **median** instead,
because a few replications have very large resampled spline SEs.

**Coverage in the subtitle** is computed from the per-replication bands before
averaging: *pointwise cov* is the fraction of replications whose own band
contains the true value at each grid point (averaged over the grid; minimum in
parentheses; target ≈ 0.95), and *sim* is the fraction of replications whose
band contains the entire true curve simultaneously.

---

## Reproducing the simulation study

```bash
# list the 7 canonical settings (cox/aft/npmle/ltm + three frailty variants)
python -m ode_unify.simulate_study list

# run all settings (100 reps each) and draw the band plots
python -m ode_unify.simulate_study all --reps 100 --workers 9

# one setting / redraw plots only
python -m ode_unify.simulate_study run  --only aft_setting2 --reps 100
python -m ode_unify.simulate_study plot --only aft_setting2
```

Per-replication results land in `ode_unify/results/<setting>/` and the figures
in `ode_unify/plots/` (`cox_s1`, `aft_s2`, `npmle_s3`, `re_cox_s1`,
`re_aft_s2`, `ltm_s4_alpha`, `ltm_s4_q`, `re_ltm_s1_alpha`, `re_ltm_s1_q`).

## Reproducibility

`ode_unify` reproduces the standalone per-model reference code **exactly**:

```bash
python -m ode_unify.sanity_check            # all 7 (estimator, frailty) combos
python -m ode_unify.sanity_check --only cox re_cox
```

Each check fits the same dataset through the unified pipeline and through the
original `RecurrentODE_py` `main()`/`inference()` and compares point estimates
and SEs — all 7 combos match to machine precision (0.0). The full 7×100
simulation study rerun through `ode_unify` reproduced every one of the 700
per-replication point-estimate files **bit-for-bit**.

---

## A note on the MIMIC data

The `mimic_analysis/` scripts apply the method to the **MIMIC-III** clinical
database. MIMIC-III is credentialed, proprietary data and is **not**
redistributed here — neither the raw data nor patient-level derived files are
included. To reproduce that analysis, obtain MIMIC-III access
([physionet.org](https://physionet.org/content/mimiciii/)) and regenerate the
inputs locally. Only aggregate result plots are provided.

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
